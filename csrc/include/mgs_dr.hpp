/* mgs_dr.hpp
 *
 * contains declarations for the top-level rendering function, interfacing with pytorch
 */

#ifndef MGS_DR_H
#define MGS_DR_H

#include <ATen/Operators.h>
#include <torch/all.h>
#include <tuple>

#include "mgs_dr_global.h"

//-------------------------------------------//

class MGSDRsettingsTorch : public torch::CustomClassHolder
{
public:
	MGSDRsettingsTorch(int64_t width, int64_t height, const at::Tensor& view, const at::Tensor& proj,
	                   double focalX, double focalY, bool debug);

	MGSDRsettings settings;
};

//-------------------------------------------//

std::tuple<at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_forward(const c10::intrusive_ptr<MGSDRsettingsTorch>& settings,
               const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_backward(const c10::intrusive_ptr<MGSDRsettingsTorch>& settings, const at::Tensor& dLdImage,
                const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics,
                int64_t numRendered, const at::Tensor& geomBufs, const at::Tensor& binningBufs, const at::Tensor& imageBufs);

#endif //#ifndef MGS_DR_H
