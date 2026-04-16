cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(MRPNN LANGUAGES CXX CUDA)

enable_language(CUDA)

IF (WIN32)

MESSAGE(STATUS "Is windows")
SET(CMAKE_CXX_FLAGS "/openmp /Ot")

ELSE ()

MESSAGE(STATUS "Is linux")
add_definitions(-DLINUX)

ENDIF ()

Option(GUI OFF)
Option(RTX30XX OFF)
Option(TENSORCORE OFF)
Option(CompileRPNN OFF)
Option(DISABLE_FAST_MATH OFF)

# =========================
# Python / Torch / pybind11
# =========================
# Assumes local venv at training_pipeline/.venv relative to this CMakeLists
#set(PYTHON_VENV_ROOT "${CMAKE_SOURCE_DIR}/training_pipeline/.venv")

#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

## Python include / libs from the venv
#set(PYTHON_INCLUDE_DIR "${PYTHON_VENV_ROOT}/Include")
#set(PYTHON_LIBRARY_DIR "${PYTHON_VENV_ROOT}/libs")
#
## Python 3.13
#set(PYTHON_LIB_NAME "python313")
#
## site-packages
#set(PY_ENV_SITE_PACKAGES "${PYTHON_VENV_ROOT}/Lib/site-packages")
#
## Torch headers and libs from pip wheel
#set(TORCH_INCLUDE_DIRS
#    "${PY_ENV_SITE_PACKAGES}/torch/include"
#    "${PY_ENV_SITE_PACKAGES}/torch/include/torch/csrc/api/include"
#)
#set(TORCH_LIB_DIR "${PY_ENV_SITE_PACKAGES}/torch/lib")
#
## pybind11 headers from pip wheel
#set(PYBIND11_INCLUDE_DIR "${PY_ENV_SITE_PACKAGES}/pybind11/include")
#
## =========================

# Python / Torch paths
set(PYTHON_VENV_ROOT "${CMAKE_SOURCE_DIR}/training_pipeline/.venv_cp312")
set(PY_ENV_SITE_PACKAGES "${PYTHON_VENV_ROOT}/Lib/site-packages")

# Torch headers and libs from pip wheel
set(TORCH_INCLUDE_DIRS
    "${PY_ENV_SITE_PACKAGES}/torch/include"
    "${PY_ENV_SITE_PACKAGES}/torch/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR "${PY_ENV_SITE_PACKAGES}/torch/lib")

# pybind11 headers:
# you can either use Torch's bundled pybind11...
set(PYBIND11_INCLUDE_DIR "${PY_ENV_SITE_PACKAGES}/torch/include")
# ...or, if you pip install pybind11, you could instead do:
# set(PYBIND11_INCLUDE_DIR "${PY_ENV_SITE_PACKAGES}/pybind11/include")
set(Python3_INCLUDE_DIRS, "C:/Users/five1/AppData/Local/Programs/Python/Python312/libs")

if (GUI)
    add_definitions(-DGUI)
    find_package(GLEW)
    find_package(glfw3)
    add_subdirectory(Imgui)
    INCLUDE_DIRECTORIES(Imgui_INCLUDE_DIR)
endif()

if (CompileRPNN)
    add_definitions(-DCRPNN)
endif()

if (TENSORCORE)
    add_definitions(-DENBALE_TENSOR)
endif()


file(GLOB CORE_HEADERS core/*.h *core/*.hpp core/*.cuh)
file(GLOB CORE_SOURCES core/*.cpp core/*.cu)

source_group("Core Include" FILES ${CORE_HEADERS}) 
source_group("Core Source" FILES ${CORE_SOURCES})

include_directories(${PROJECT_SOURCE_DIR}/core)
  
add_executable(DataGen ${CORE_HEADERS} ${CORE_SOURCES} dataGen.cu)
set_target_properties(DataGen PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(DataGen PRIVATE _HAS_STD_BYTE=0)

add_executable(VolumeRender ${CORE_HEADERS} ${CORE_SOURCES} main.cu)
set_target_properties(VolumeRender PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(VolumeRender PRIVATE _HAS_STD_BYTE=0)

add_executable(Test ${CORE_HEADERS} ${CORE_SOURCES} test.cu)
set_target_properties(Test PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CXX_STANDARD_REQUIRED ON
CXX_STANDARD 17)
target_compile_definitions(Test PRIVATE _HAS_STD_BYTE=0)

# ===========================================
# Python extension module: mrpnn_input_data_extension
# ===========================================
set(PY_EXTENSION_SOURCES
    mrpnn_input_data_wrapper.cpp
    mrpnn_input_data_kernel.cu
)

add_library(mrpnn_input_data_extension MODULE
    ${CORE_HEADERS}
    ${CORE_SOURCES}
    ${PY_EXTENSION_SOURCES}
)

set_target_properties(mrpnn_input_data_extension PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
    SUFFIX ".pyd"
)

target_compile_definitions(mrpnn_input_data_extension PRIVATE
    _HAS_STD_BYTE=0
    TORCH_API_INCLUDE_EXTENSION_H
    TORCH_EXTENSION_NAME=mrpnn_input_data_extension   # must match your PYBIND11_MODULE name
)

target_include_directories(mrpnn_input_data_extension PRIVATE
    ${PROJECT_SOURCE_DIR}/core
    ${Python3_INCLUDE_DIRS}      # Python.h from vcpkg Python
    ${TORCH_INCLUDE_DIRS}        # torch/extension.h etc. from venv
    ${PYBIND11_INCLUDE_DIR}      # pybind11/pybind11.h
)

target_link_directories(mrpnn_input_data_extension PRIVATE
    ${TORCH_LIB_DIR}             # where torch.lib, c10.lib, etc live
)

target_link_libraries(mrpnn_input_data_extension PRIVATE
    python312              		# CPython import lib from vcpkg
    c10
    torch
    torch_cpu
    torch_cuda                   # if present
	torch_python
)

# =======================
# CUDA arch / fast math
# =======================
if (DISABLE_FAST_MATH)
    if (RTX30XX)
        target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
        target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
        target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
        target_compile_options(mrpnn_input_data_extension PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_86 -maxrregcount=168>)
    else()
        target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
        target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
        target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
        target_compile_options(mrpnn_input_data_extension PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -maxrregcount=168>)
    endif()
else()
    if (RTX30XX)
        target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
        target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
        target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
        target_compile_options(mrpnn_input_data_extension PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_86 -maxrregcount=168>)
    else()
        target_compile_options(VolumeRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
        target_compile_options(Test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
        target_compile_options(DataGen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
        target_compile_options(mrpnn_input_data_extension PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math -arch=sm_75 -maxrregcount=168>)
    endif()
endif()

if (GUI)
    target_link_libraries(VolumeRender GLEW::GLEW glfw Imgui) 
    add_dependencies(VolumeRender Imgui)
    target_link_libraries(Test GLEW::GLEW glfw Imgui) 
    add_dependencies(Test Imgui)
    target_link_libraries(DataGen GLEW::GLEW glfw Imgui) 
    add_dependencies(DataGen Imgui)
endif()
