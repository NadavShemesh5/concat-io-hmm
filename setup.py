from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup


setup(ext_modules=[Pybind11Extension("algo.baum_welch", ["algo/baum_welch.cpp"], cxx_std=11),
                   Pybind11Extension("algo.io_baum_welch", ["algo/io_baum_welch.cpp"], cxx_std=11)])
