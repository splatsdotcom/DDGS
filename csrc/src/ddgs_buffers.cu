#define __FILENAME__ "ddgs_forward.cu"

#include "ddgs_buffers.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "ddgs_global.h"

//-------------------------------------------//

//TODO: should we pass debug here and validate the empty sum/sort calls?

DDGSrenderBuffers::DDGSrenderBuffers(uint8_t* mem, uint32_t count) :
	m_mem(mem), m_count(count)
{

}

DDGSgeomBuffers::DDGSgeomBuffers(uint8_t* mem, uint32_t count) :
	DDGSrenderBuffers(mem, count)
{
	pixCenters       = bump<QMvec2>();
	pixRadii         = bump<float>();
	depths           = bump<float>();
	tilesTouched     = bump<uint32_t>();
	covs             = bump<DDGScov3D>();
	conicOpacity     = bump<QMvec4>();
	color            = bump<QMvec3>();
	tilesTouchedScan = bump<uint32_t>();

	cub::DeviceScan::InclusiveSum(nullptr, tilesTouchedScanTempSize, tilesTouched, tilesTouchedScan, count);
	tilesTouchedScanTemp = bump<uint8_t>(tilesTouchedScanTempSize);
}

DDGSbinningBuffers::DDGSbinningBuffers(uint8_t* mem, uint32_t count) :
	DDGSrenderBuffers(mem, count)
{
	keys          = bump<uint64_t>();
	indices       = bump<uint32_t>();
	keysSorted    = bump<uint64_t>();
	indicesSorted = bump<uint32_t>();

	cub::DeviceRadixSort::SortPairs(nullptr, sortTempSize, keys, keysSorted, indices, indicesSorted, count);
	sortTemp = bump<uint8_t>(sortTempSize);
}

DDGSimageBuffers::DDGSimageBuffers(uint8_t* mem, uint32_t count) :
	DDGSrenderBuffers(mem, count)
{
	//TODO: this is wasteful! only need 1 per tile, can also get away with just a uint32_t
	tileRanges      = bump<uint2>();
	transmittance   = bump<float>();
	numContributors = bump<uint32_t>();
}

DDGSderivativeBuffers::DDGSderivativeBuffers(uint8_t* mem, uint32_t count) :
	DDGSrenderBuffers(mem, count)
{
	dLdPixCenters = bump<QMvec2>();
	dLdConics     = bump<QMvec3>();
	dLdColors     = bump<QMvec3>();
}