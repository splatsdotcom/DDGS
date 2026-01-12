/* ddgs.hpp
 *
 * contains declarations for the top-level rendering function, interfacing with pytorch
 */

#ifndef DDGS_H
#define DDGS_H

#include <ATen/Operators.h>
#include <torch/all.h>
#include <tuple>

#include "ddgs_global.h"

//-------------------------------------------//

class DDGSsettingsTorch : public torch::CustomClassHolder
{
public:
	DDGSsettingsTorch(int64_t width, int64_t height, const at::Tensor& view, const at::Tensor& proj,
	                   double focalX, double focalY, bool debug);

	DDGSsettings settings;
};

//-------------------------------------------//

std::tuple<at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor>
ddgs_forward(const c10::intrusive_ptr<DDGSsettingsTorch>& settings,
             const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ddgs_backward(const c10::intrusive_ptr<DDGSsettingsTorch>& settings, const at::Tensor& dLdImage,
              const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics,
              int64_t numRendered, const at::Tensor& geomBufs, const at::Tensor& binningBufs, const at::Tensor& imageBufs);

#endif //#ifndef DDGS_H
