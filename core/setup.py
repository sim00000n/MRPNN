# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='radiance_extension',
    ext_modules=[
        CUDAExtension(
            name='radiance_extension',
            sources=[
                'mrpnn_input_data_kernel.cu',
                'mrpnn_input_data_wrapper.cpp',
                'camera.cu',
                'radiancePredict.cu',
                'renderResources.cu',
                'vector.cu',
                'volume.cu'
            ],
            # Optional: Add compiler optimizations
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-D_HAS_STD_BYTE=0']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)