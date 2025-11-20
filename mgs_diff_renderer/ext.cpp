#include <Python.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include "mgs_dr.hpp"

//-------------------------------------------//

//dummy module iniitalization
extern "C" {
	PyObject* PyInit__C(void)
	{
		static struct PyModuleDef module_def = {
			PyModuleDef_HEAD_INIT,
			"_C",
			NULL,
			-1,
			NULL,
		};
		return PyModule_Create(&module_def);
	}
}

TORCH_LIBRARY(mgs_diff_renderer, m) 
{
	m.class_<MGSDRsettingsTorch>("Settings")
		.def(torch::init<int64_t, int64_t, const at::Tensor&, const at::Tensor&, double, double, bool>());

	m.def(
		"forward(__torch__.torch.classes.mgs_diff_renderer.Settings settings, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics) -> (Tensor, int, Tensor, Tensor, Tensor)",
		&mgs_dr_forward
	);
	m.def(
		"backward(__torch__.torch.classes.mgs_diff_renderer.Settings settings, Tensor dLdImage, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics, int numRendered, Tensor geomBufs, Tensor binningBufs, Tensor imageBufs) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
		&mgs_dr_backward
	);
}
