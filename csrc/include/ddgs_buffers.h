/* ddgs_buffers.h
 *
 * contains declarations for the temp buffers 
 * used in the differentiable rendering process
 */

#ifndef DDGS_BUFFERS_H
#define DDGS_BUFFERS_H

#include <math.h>
#include <stdint.h>
#include "ddgs_global.h"

//-------------------------------------------//

#define DDGS_L1_CACHE_ALIGNMENT 128

//-------------------------------------------//

class DDGSrenderBuffers
{
public:
	DDGSrenderBuffers(uint8_t* mem, uint32_t count);

	template<typename T>
	static uint64_t required_mem(uint32_t count)
	{
		T buffer(nullptr, count);
		return reinterpret_cast<uint64_t>(buffer.m_mem);
	}
	
private:
	uint8_t* m_mem;
	uint32_t m_count;

protected:
	template<typename T>
	T* bump()
	{
		uint64_t alignment = std::max((uint64_t)DDGS_L1_CACHE_ALIGNMENT, (uint64_t)sizeof(T));
		uint64_t offset = (reinterpret_cast<uintptr_t>(m_mem) + alignment - 1) & ~(alignment - 1);
		
		T* ptr = reinterpret_cast<T*>(offset);
		m_mem = reinterpret_cast<uint8_t*>(ptr + m_count);

		return ptr;
	}

	template<typename T>
	T* bump(uint64_t size)
	{
		uint64_t alignment = std::max((uint64_t)DDGS_L1_CACHE_ALIGNMENT, (uint64_t)sizeof(T));
		uint64_t offset = (reinterpret_cast<uintptr_t>(m_mem) + alignment - 1) & ~(alignment - 1);
		
		T* ptr = reinterpret_cast<T*>(offset);
		m_mem = reinterpret_cast<uint8_t*>(ptr) + size;

		return ptr;
	}
};

struct DDGSgeomBuffers : public DDGSrenderBuffers
{
public:
	DDGSgeomBuffers(uint8_t* mem, uint32_t count);

	QMvec2*    __restrict__ pixCenters;
	float*     __restrict__ pixRadii;
	float*     __restrict__ depths;
	uint32_t*  __restrict__ tilesTouched;
	uint32_t*  __restrict__ tilesTouchedScan;
	DDGScov3D* __restrict__ covs;
	QMvec4*    __restrict__ conicOpacity;
	QMvec3*    __restrict__ color;

	size_t tilesTouchedScanTempSize;
	uint8_t* __restrict__ tilesTouchedScanTemp;
};

struct DDGSbinningBuffers : public DDGSrenderBuffers
{
public:
	DDGSbinningBuffers(uint8_t* mem, uint32_t count);

	uint64_t* __restrict__ keys;
	uint32_t* __restrict__ indices;
	uint64_t* __restrict__ keysSorted;
	uint32_t* __restrict__ indicesSorted;

	size_t sortTempSize;
	uint8_t* sortTemp;
};

struct DDGSimageBuffers : public DDGSrenderBuffers
{
public:
	DDGSimageBuffers(uint8_t* mem, uint32_t count);

	uint2*    __restrict__ tileRanges;
	float*    __restrict__ transmittance;
	uint32_t* __restrict__ numContributors;
};

struct DDGSderivativeBuffers : public DDGSrenderBuffers
{
public:
	DDGSderivativeBuffers(uint8_t* mem, uint32_t count);

	QMvec2* __restrict__ dLdPixCenters;
	QMvec3* __restrict__ dLdConics;
	QMvec3* __restrict__ dLdColors;
};

#endif //#ifndef DDGS_BUFFERS_H