/* ddgs_global.h
 *
 * contains utility functions, macros, and global config
 * for the differentiable renderer
 */

#ifndef DDGS_GLOBAL_H
#define DDGS_GLOBAL_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

#define QM_FUNC_ATTRIBS __device__ static inline
#include "../external/QuickMath/quickmath.h"

// only enable for personal testing:
// #define DDGS_PROFILE

#define DDGS_TILE_SIZE 16
#define DDGS_TILE_LEN (DDGS_TILE_SIZE * DDGS_TILE_SIZE)

#define DDGS_MAX_ALPHA 0.99f
#define DDGS_MIN_ALPHA (1.0f / 255.0f)
#define DDGS_ACCUM_ALPHA_CUTOFF 0.00001f

//-------------------------------------------//

struct DDGSgaussians
{
	uint32_t count;

	float* __restrict__ means;
	float* __restrict__ scales;
	float* __restrict__ rotations;
	float* __restrict__ opacities;
	float* __restrict__ harmonics;
};

struct DDGSsettings
{
	uint32_t width;
	uint32_t height;

	QMmat4 view;
	QMmat4 proj;
	QMmat4 viewProj;

	float focalX;
	float focalY;

	bool debug;
};

struct DDGScov3D
{
	float m00, m01, m02;
	float      m11, m12;
	float           m22;
};

//-------------------------------------------//

__device__ __host__ static __forceinline__ uint32_t _ddgs_ceildivide32(uint32_t a, uint32_t b)
{
	return (a + b - 1) / b;
}

//-------------------------------------------//

#define DDGS_CUDA_ERROR_CHECK(s)                                        \
	s;                                                                    \
	if(settings.debug)                                                    \
	{                                                                     \
		cudaError error = cudaDeviceSynchronize();                        \
		if(error != cudaSuccess)                                          \
		{                                                                 \
			std::cerr << std::endl << "DDGS: CUDA error in \"" <<        \
				__FILENAME__ << "\" at line " << __LINE__ <<              \
				": \"" << cudaGetErrorString(error) << "\"" << std::endl; \
			throw std::runtime_error("DDGS CUDA error");                 \
		}                                                                 \
	}

#ifdef DDGS_PROFILE
	#define DDGS_PROFILE_REGION_START(name) cudaDeviceSynchronize(); auto tStart##name = std::chrono::high_resolution_clock::now()
	#define DDGS_PROFILE_REGION_END(name) cudaDeviceSynchronize(); auto tEnd##name = std::chrono::high_resolution_clock::now()

	#define DDGS_PROFILE_REGION_TIME(name) std::chrono::duration_cast<std::chrono::microseconds>(tEnd##name - tStart##name).count() / 1000.0
#else
	#define DDGS_PROFILE_REGION_START(name)
	#define DDGS_PROFILE_REGION_END(name)

	#define DDGS_PROFILE_REGION_TIME(name) 0.0
#endif

#endif //#ifndef DDGS_GLOBAL_H
