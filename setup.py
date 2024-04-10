import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup_dir = os.path.dirname(os.path.realpath(__file__))

ext_modules = [
    Pybind11Extension(
        "libsegmenter.bindings",
        ["libsegmenter/SegmenterBindings.cpp"],
        include_dirs=[
            os.path.join(setup_dir, "libsegmenter/"),
        ],
        extra_compile_args=["-O3", "-std=c++20"],
    ),
]

setup(
    name="libsegmenter",
    version="0.1",
    packages=find_packages(),  # Ensure all packages are found
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_package_data=True,  # Include files specified in MANIFEST.in
)
