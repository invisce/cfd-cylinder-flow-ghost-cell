import sys
from setuptools import setup, Extension
import pybind11

is_win = sys.platform == 'win32'

if is_win:
    extra_compile_args = ['/O2', '/std:c++17', '/fp:precise', '/openmp']
    extra_link_args = []
else:
    extra_compile_args = ['-O3', '-std=c++17', '-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        'mac_engine',
        ['src/bindings.cpp'],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    name='mac_engine',
    ext_modules=ext_modules,
    zip_safe=False,
)
