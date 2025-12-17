/* ddgs_forward.h
 *
 * contains declarations for the forward rendering functions,
 * implemented in CUDA
 */

#ifndef DDGS_FORWARD_H
#define DDGS_FORWARD_H

#include <functional>
#include <stdint.h>
#include "ddgs_global.h"

typedef std::function<uint8_t* (uint64_t size)> DDGSresizeFunc;

//-------------------------------------------//

uint32_t ddgs_forward_cuda(DDGSsettings settings, DDGSgaussians gaussians,
                           DDGSresizeFunc createGeomBuf, DDGSresizeFunc createBinningBuf, DDGSresizeFunc createImageBuf,
                           float* outImg);

#endif //#ifndef DDGS_FOWARD_H