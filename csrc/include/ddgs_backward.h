/* ddgs_backward.h
 *
 * contains declataions for the backward rendering pass,
 * implemented in CUDA
 */

#ifndef DDGS_BACKWARD_H
#define DDGS_BACKWARD_H

#include <stdint.h>
#include "ddgs_global.h"

//-------------------------------------------//

void ddgs_backward_cuda(DDGSsettings settings, const float* dLdImage, DDGSgaussians gaussians,
                        uint32_t numRendered, const uint8_t* geomBufsMem, const uint8_t* binningBufsMem, const uint8_t* imageBufsMem,
                        DDGSgaussians outDLdGaussians);

#endif //#ifndef DDGS_BACKWARD_H