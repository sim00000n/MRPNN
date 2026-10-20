// **radiance_wrapper.cpp** (Stateful version)

//#ifdef _DEBUG
//    #pragma push_macro("_DEBUG")
//    #undef _DEBUG
//    #include <torch/extension.h>
//    #include <pybind11/pybind11.h> // Include pybind11 for class binding
//    #pragma pop_macro("_DEBUG")
//#else
//#undef _DEBUG

    #include <torch/extension.h>
    #include <pybind11/pybind11.h> // Include pybind11 for class binding
//#endif

#include <cuda_runtime.h>
#include "volume.hpp"
#include "camera.hpp"
#include <array>

//#pragma optimize("",off)

namespace py = pybind11;

// Declare the launcher function from the .cu file (possibly accepting state data)
void fetch_input_kernel_launcher(
    float* __restrict__ out_buffer,       // W * H * 2304 buffer (pre-allocated on GPU)
    Camera& cam,
    float3 lightDir,
    float alpha,
    float g,
    int2 pixelOffset);

class MRPNNInputData {
public:
    torch::Tensor weights;
    //int mem_footprint = 192 * 3 + 1;
    int mem_footprint = 192 * 3 + 5 + 64 * 3;


    // --- CONSTRUCTOR ---
    // This runs when you call RadiancePredictor() in Python
    MRPNNInputData(int width, int height, string vol_path) :
        current_path(vol_path),
        m_volumeRender(vol_path),
        m_cam(m_volumeRender, "test camera")
    {
        //static volatile bool debug_break = true;
        //while (debug_break);

        // **YOUR STATE INITIALIZATION LOGIC GOES HERE**
        // Example: Allocate a dummy weights tensor and move it to CUDA
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA required for MRPNNInputData.");
        }
        
        width /= 8;
        height /= 8;

        m_cam.resolution = width;
        
        float3 scatter = float3{ 1, 1, 1 };
        m_volumeRender.SetScatterRate(scatter);

        m_tensor = torch::empty(
            { height, width, mem_footprint },
            torch::TensorOptions(torch::kFloat32).device(torch::kCUDA)
        );
    }

    // --- STATEFUL METHOD (The new routine) ---
    // This is the function Python will call repeatedly
    torch::Tensor fetch_input(float3 camPos = float3{ 0.67085f, -0.038080f, -0.04856f }, float3 lightDir = normalize(float3{ 0.34281f, 0.70711f, 0.61845f }), float alpha = 1.0f, float g = 0.857f, int2 pixelOffset = { 0, 0 })
    {
        m_cam.SetPosition(camPos);

        m_volumeRender.UpdateHGLut(g);

        // Launch kernel, passing the state member (weights_)
        fetch_input_kernel_launcher(m_tensor.data_ptr<float>(), m_cam, lightDir, alpha, g, pixelOffset);

        return m_tensor;
    }

    void swap_data(string vol_path)
    {
        if (current_path != vol_path)
        {
            m_volumeRender.LoadData(vol_path, false);
            current_path = vol_path;
        }
    }

private:
    VolumeRender m_volumeRender;
    Camera m_cam;
    torch::Tensor m_tensor;
    string current_path;
};

// --- PYBIND11 BINDING ---
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    py::class_<MRPNNInputData>(m, "MRPNNInputData")
//        .def(py::init<int, int>(),
//            py::arg("width"),
//            py::arg("height"))
//        .def("fetch_input", &MRPNNInputData::fetch_input,
//            "Fetches the NN inference input using optional camera and light parameters.",
//            py::arg("camPos") = float3{ 0.67085f, -0.03808f, -0.04856f },
//            py::arg("lightDir") = float3{ 0.34281f, 0.70711f, 0.61845f }, // Replace with actual normalized value
//            py::arg("g") = 0.857f
//        );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MRPNNInputData>(m, "MRPNNInputData")
        .def(py::init<int, int, string>(),
            py::arg("width"),
            py::arg("height"),
            py::arg("vol_path"))
        .def("fetch_input",
            [](MRPNNInputData& self,
                std::array<float, 3> camPos,
                std::array<float, 3> lightDir,
                float alpha,
                float g,
                std::array<int, 2> pixelOffset) {
                    float3 cam{
                        camPos[0],
                        camPos[1],
                        camPos[2]
                    };
                    float3 light{
                        lightDir[0],
                        lightDir[1],
                        lightDir[2]
                    };
                    light = normalize(light);
                    int2 po{
                        pixelOffset[0],
                        pixelOffset[1]
                    };

                    try {
                        return self.fetch_input(cam, light, alpha, g, po);
                    }
                    catch (const c10::Error& e) {
                        throw std::runtime_error(std::string("c10::Error: ") + e.what());
                    }
                    catch (const std::exception& e) {
                        throw std::runtime_error(std::string("C++ exception: ") + e.what());
                    }
                    catch (...) {
                        throw std::runtime_error("Unknown non-std exception in fetch_input");
                    }
            },
            "Fetches the NN inference input using optional camera and light parameters.",
            py::arg("camPos") = std::array<float, 3>{ 0.67085f, -0.03808f, -0.04856f },
            py::arg("lightDir") = std::array<float, 3>{ 0.34281f, 0.70711f, 0.61845f  },
            py::arg("alpha") = 1.0f,
            py::arg("g") = 0.857f,
            py::arg("pixelOffset") = std::array<int, 2>{ 0, 0 }
        )
        .def("swap_data",
            &MRPNNInputData::swap_data,
            py::arg("vol_path"),
            "Loads new volume data."
        );
}