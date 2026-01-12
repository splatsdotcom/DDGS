#include <Python.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include "ddgs.hpp"

//-------------------------------------------//

//dummy module iniitalization
extern "C" {
	PyObject* PyInit__C(void)
	{
		static struct PyModuleDef moduleDef = {
			PyModuleDef_HEAD_INIT,
			"_C",
			NULL,
			-1,
			NULL,
		};
		return PyModule_Create(&moduleDef);
	}
}

TORCH_LIBRARY(ddgs, m) 
{
	m.class_<DDGSsettingsTorch>("Settings")
		.def(torch::init<int64_t, int64_t, const at::Tensor&, const at::Tensor&, double, double, bool>());

	m.def(
		"forward(__torch__.torch.classes.ddgs.Settings settings, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics) -> (Tensor, Tensor, int, Tensor, Tensor, Tensor)",
		&ddgs_forward
	);
	m.def(
		"backward(__torch__.torch.classes.ddgs.Settings settings, Tensor dLdImage, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics, int numRendered, Tensor geomBufs, Tensor binningBufs, Tensor imageBufs) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
		&ddgs_backward
	);
}
