from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "src/advanced_search_cpp",
        ["src/advanced_search_module.cpp", "src/advanced_search.cpp"],
        include_dirs=[".", "<path_to_pybind11_include>"],
        extra_compile_args=['-std=c++11', '-Ofast'],
    ),
]

setup(
    name="src/advanced_search_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)