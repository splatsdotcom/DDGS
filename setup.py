import os
import torch
import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
	CUDAExtension,
	BuildExtension
)

# ------------------------------------------- #

LIBRARY_NAME = "mgs_diff_renderer"
LIBRARY_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), LIBRARY_NAME)
CSRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "csrc")

if torch.__version__ >= "2.6.0": # TODO: what is this ?
	limitedAPI = True
else:
	limitedAPI = False

# ------------------------------------------- #

def get_extension():
	linkArgs = [
		"-O3"
	]
	compileArgs = {
		"cxx": [
			"-O3",
			"-DPy_LIMITED_API=0x03090000",
		],
		"nvcc": [
			"-O3",
		],
	}

	srcDir = os.path.join(CSRC_DIR, "src")
	srcs = list((
		glob.glob(os.path.join(srcDir, "*.cpp")) +
		glob.glob(os.path.join(srcDir, "*.cu"))
	))
	srcs.append(
		os.path.join(LIBRARY_DIR, "ext.cpp")
	)

	includeDirs = [
		os.path.join(CSRC_DIR, "include"),
		os.path.join(CSRC_DIR, "external")
	]

	return CUDAExtension(
		f"{LIBRARY_NAME}._C",
		srcs,
		include_dirs=includeDirs,
		extra_compile_args=compileArgs,
		extra_link_args=linkArgs,
		py_limited_api=limitedAPI
	)

# ------------------------------------------- #

setup(
	packages=find_packages(),
	ext_modules=[ get_extension() ],
	cmdclass={"build_ext": BuildExtension},
	options={"bdist_wheel": {"py_limited_api": "cp39"}} if limitedAPI else {},
)
