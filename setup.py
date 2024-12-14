from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
here = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    Pybind11Extension(
        "advanced_search_cpp",
        [
            os.path.join("src", "module.cpp"),
            os.path.join("src", "base_search.cpp"),
            os.path.join("src", "linear_search.cpp"),
            os.path.join("src", "knn_search.cpp"),
        ],
        include_dirs=[os.path.join(here, "include")],
        extra_compile_args=[
            '-std=c++17', '-Ofast', '-march=native', '-funroll-loops',
            '-fprefetch-loop-arrays', '-ffast-math', '-flto', '-fopenmp'
        ],
        extra_link_args=['-fopenmp'],
    ),
]

def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().splitlines()

requirements = parse_requirements("requirements.txt")

setup(
    name="advanced_search_cpp", 
    version="0.1.0",  
    description="A high-performance advanced vector search package with C++ and Python integration.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="SUNGOD",
    author_email="sun1223god@gmail.com",
    url="https://github.com/SUNGOD3/AdvancedVectorSearch",
    license="MIT",  
    packages=find_packages(where="src"),
    package_dir={"": "src"}, 
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    options={
        "bdist_wheel": {
            "plat_name": "manylinux2014_x86_64",  
        },
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  
    install_requires=requirements,
    zip_safe=False,  
)
