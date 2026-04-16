#include "volume.hpp"
#include "camera.hpp"
#include "sample_method.hpp"
#include "GUI.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include <filesystem>
#include <time.h>

float CompareBias(string path, string name, string path2, string name2) {
#if !LINUX
    wchar_t szCmdLineW[500] = { 0 };
    char buffer[500] = { 0 };
    DWORD bytesRead = 0;
    string str = "cmd.exe /c python \"./tools/Compare.py\" \"";
    str += path + "\" \"" + name + "\" \"" + path2 + "\" \"" + name2 + "\"";
    size_t convertedChars = 0;
    mbstowcs_s(&convertedChars, szCmdLineW, str.length() + 1, str.c_str(), _TRUNCATE);
    SECURITY_ATTRIBUTES sa = { 0 };
    HANDLE hRead = NULL, hWrite = NULL;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0))
        return -1;
    STARTUPINFOW si = { 0 };
    PROCESS_INFORMATION pi = { 0 };
    si.cb = sizeof(STARTUPINFO);
    GetStartupInfoW(&si);
    si.hStdError = hWrite;
    si.hStdOutput = hWrite;
    si.wShowWindow = SW_HIDE;
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    if (!CreateProcessW(NULL, szCmdLineW, NULL, NULL, TRUE, NULL, NULL, NULL, &si, &pi)) {
        CloseHandle(hWrite);
        CloseHandle(hRead);
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(hWrite);
    ReadFile(hRead, buffer, 500, &bytesRead, NULL);
    CloseHandle(hRead);
    float bias = atof(buffer);
    return bias;
#else
    reutrn 0;
#endif
}

float3 GetRandVec(float min, float max, bool isSqLerp = false)
{
    float3 retVal;
    retVal.x = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    retVal.y = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    retVal.z = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    float lerpVal = (float)rand() / (float)RAND_MAX;
    if (isSqLerp)
        lerpVal *= lerpVal;

    float length = lerp(min, max, lerpVal);

    return normalize(retVal) * length;
}

int main()
{
    srand(time(0));

    string GT_path = "./Results/GT/";
    string predict_path = "./Results/";
    string test_path = "./TestCase/";

    int sample_num = 256;
    VolumeRender::RenderType RunType = VolumeRender::RenderType::PT;

    string Log_path = RunType == VolumeRender::RenderType::RPNN ? "./Log_RPNN.txt" : "./Log_MRPNN.txt";

    predict_path += RunType == VolumeRender::RenderType::RPNN ? "RPNN/" : "MRPNN/";

    std::filesystem::create_directories(predict_path);

    const int test_num = 4;
    string names[test_num] = { "CLOUD0", "CLOUD1",  "MODEL0", "MODEL1"};
    float3 lightDirs[test_num][3] = { 
        {
            normalize(float3{ 0.34281, 0.70711, 0.61845 }), 
            normalize(float3{ 0.98528, 0.06976, 0.15605 }), 
            normalize(float3{ -0.90329, 0.34202, 0.55901}),
        },
        {
            normalize(float3{ -0.11803, 0.52992, 0.83979 }),
            normalize(float3{ 0.55721, 0.70711, 0.43534 }),
            normalize(float3{ -0.94026, -0.20791, -0.26961}),
        },
        {
            normalize(float3{ 0.20489, 0.87462, -0.43939 }),
            normalize(float3{ 1, 0, 0 }),
            normalize(float3{ -1,0,0 }),
        },
        {
            normalize(float3{ 0.98515, 0.12187, 0.12096 }),
            normalize(float3{ 0.39785, 0.20791, 0.89358 }),
            normalize(float3{ 0.05154, -0.17365, -0.98346 }),
        }
    };
    float3 CamPoss[test_num] = {
        float3{ 0.67085, -0.03808, -0.04856 }, float3{ 0.55471, 0.30303, 0.10048 }, 
        float3{ 0.8, 0, 0 }, float3{ -0.01786, 0.11180, 0.56642 }
    };
    float alphas[test_num] = { 2.0f,2.0f,2.0f,4.0f };

    FILE* log = fopen(Log_path.c_str(), "w");
    float2 avrg = { 0 };
    int start_time = clock();

    int id = 0;
    string name = names[0];
    VolumeRender v(test_path + name, true);

    float3 scatter = float3{ 1.0f, 1.0f, 1.0f };
    v.SetScatterRate(scatter);

    Camera cam(v, "test camera");
    cam.resolution = 512;
    float g = 0.857f;
    float3 lightColor = { 1.0, 1.0, 1.0 };

    bool RenderSpecific = false;

    if (RenderSpecific)
    {
        string name = "MODEL0";
        v.LoadData(test_path + name, true);

        float3 CamPos = { -0.056679f, 0.105890f, -0.308501f };
        float3 lightDir = { 0.565041f, -0.531418f, 0.631129f };
        float alpha = 0.825488f;
        cam.SetPosition(CamPos);
        string filename = name + "_C" + std::to_string(CamPos.x) + "_" + std::to_string(CamPos.y) + "_" + std::to_string(CamPos.z) + "_L" + std::to_string(lightDir.x) + "_" + std::to_string(lightDir.y) + "_" + std::to_string(lightDir.z) + "_A" + std::to_string(alpha);
        cam.RenderToFile(predict_path + "/" + filename, lightDir, lightColor, alpha, 256, g, sample_num, Camera::None, RunType);

        return;
    }

    while(true)
    {
        float alpha = ((float)rand() / (float)RAND_MAX) * alphas[id];
        //float alpha = ((float)rand() / (float)RAND_MAX) * 0.5f;
        float3 lightDir = GetRandVec(1.0f, 1.0f);

        for (int j = 0; j < 2; j++)
        {
            float3 CamPos;
            
            /*float dice = rand();
            if(dice < RAND_MAX/2)
                CamPos = GetRandVec(0.0f, 1.5f);
            else
                CamPos = GetRandVec(1.5f, 10.0f);*/
            float dice = rand();
            if (dice > RAND_MAX / 2)
                CamPos = GetRandVec(0.0f, 2.0f, false);
            else
                CamPos = GetRandVec(2.0f, 10.0f, false);
            
            cam.SetPosition(CamPos);

            //string filename = name + "_" + #G + "_" + #A0 + "_" + #A1 + "_" + #A2 + "_" + dirName[j];

            //#define RunTest(G, A0, A1, A2, Type) {\
            //    float g = G;\
            //    float3 scatter = float3{ A0, A1, A2 };\
            //    string filename = name + "_C" + std::to_string(CamPos.x) + "_" + std::to_string(CamPos.y) + "_" + std::to_string(CamPos.z) + "_L" + std::to_string(lightDir.x) + "_" + std::to_string(lightDir.y) + "_" + std::to_string(lightDir.z) + "_A" + std::to_string(alpha);\
            //    v.SetScatterRate(scatter);\
            //    cam.RenderToFile(predict_path + "/" + filename, lightDir, lightColor, alpha, 256, g, sample_num, Camera::ACES, Type);\
            //    float bias = CompareBias(GT_path + name, filename, predict_path + name, filename);\
            //    printf("%s  %f\n", filename.c_str(), bias);\
            //    avrg.x += bias; avrg.y += 1;\
            //    fprintf(log, "%s  %f\n", filename.c_str(), bias);\
            //    fflush(log);\
            //}
            //
            //RunTest(0.857, 1, 1, 1, RunType);
            //if (RunType == VolumeRender::RenderType::MRPNN) {
            //    RunTest(0.5, 1, 1, 1, RunType);
            //    RunTest(0, 1, 1, 1, RunType);
            //    RunTest(0.857, 0.96, 0.98, 1, RunType);
            //    RunTest(0.857, 0.8, 0.9, 1, RunType);
            //}
            
            string filename = name + "_C" + std::to_string(CamPos.x) + "_" + std::to_string(CamPos.y) + "_" + std::to_string(CamPos.z) + "_L" + std::to_string(lightDir.x) + "_" + std::to_string(lightDir.y) + "_" + std::to_string(lightDir.z) + "_A" + std::to_string(alpha);
            cam.RenderToFile(predict_path + "/" + filename, lightDir, lightColor, alpha, 256, g, sample_num, Camera::None, RunType);
        }

        id = (id + 1) % test_num;
        name = names[id];
        v.LoadData(test_path + name, true);
    }

    printf("All Test done in  %.2f mins.\n", (clock() - start_time) / 1000.0f / 60);
    fprintf(log, "All Test done in  %.2f mins.\n", (clock() - start_time) / 1000.0f / 60);
    printf("Average bias:  %f", avrg.x / avrg.y);
    fprintf(log, "Average bias:  %f", avrg.x / avrg.y);
    fclose(log);

    return 0;
}