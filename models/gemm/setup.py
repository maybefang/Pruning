from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='gemm',
    ext_modules=[cpp_extension.CUDAExtension('gemm',['gemm.cc','gemm_kernel.cu'])],
    cmdclass={'build_ext':cpp_extension.BuildExtension})
