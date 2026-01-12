#define __FILENAME__ "ddgs_forward.cu"

#include "ddgs_forward.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "ddgs_buffers.h"
#include "ddgs_global.h"

namespace cg = cooperative_groups;

//-------------------------------------------//

//TODO: tweak these!

#define DDGS_PREPROCESS_WORKGROUP_SIZE       64
#define DDGS_KEY_WRITE_WORKGROUP_SIZE        64
#define DDGS_FIND_TILE_RANGES_WORKGROUP_SIZE 64

//-------------------------------------------//

__global__ static void __launch_bounds__(DDGS_PREPROCESS_WORKGROUP_SIZE)
_ddgs_foward_preprocess_kernel(DDGSsettings settings, DDGSgaussians gaussians, DDGSgeomBuffers outGeom);

__global__ static void __launch_bounds__(DDGS_KEY_WRITE_WORKGROUP_SIZE)
_ddgs_forward_write_keys_kernel(uint32_t width, uint32_t height,
                                uint32_t numGaussians, const DDGSgeomBuffers geom,
                                uint64_t* outKeys, uint32_t* outValues);

__global__ static void __launch_bounds__(DDGS_FIND_TILE_RANGES_WORKGROUP_SIZE)
_ddgs_forward_find_tile_ranges_kernel(uint32_t numRendered, const uint64_t* keys, uint2* outRanges);

__global__ static void __launch_bounds__(DDGS_TILE_LEN) 
_ddgs_forward_splat_kernel(DDGSsettings settings, 
                           const uint2* ranges, const uint32_t* indices, const DDGSgeomBuffers geom,
                           float* outColor, float* outAlpha, float* outTransmittance, uint32_t* outNumContributors);

__device__ static void _ddgs_get_tile_bounds(uint32_t width, uint32_t height, QMvec2 pixCenter, float pixRadius, uint2& tileMin, uint2& tileMax);

//-------------------------------------------//

uint32_t ddgs_forward_cuda(DDGSsettings settings, DDGSgaussians gaussians,
                           DDGSresizeFunc createGeomBuf, DDGSresizeFunc createBinningBuf, DDGSresizeFunc createImageBuf,
                           float* outImg, float* outAlpha)
{
	//validate:
	//---------------
	if(gaussians.count == 0)
		return 0;

	//start profiling:
	//---------------
	DDGS_PROFILE_REGION_START(total);

	//allocate geometry + image buffers:
	//---------------
	DDGS_PROFILE_REGION_START(allocateGeomImage);

	uint32_t tilesWidth  = _ddgs_ceildivide32(settings.width , DDGS_TILE_SIZE);
	uint32_t tilesHeight = _ddgs_ceildivide32(settings.height, DDGS_TILE_SIZE);
	uint32_t tilesLen = tilesWidth * tilesHeight;

	uint8_t* geomBufMem = createGeomBuf(
		DDGSrenderBuffers::required_mem<DDGSgeomBuffers>(gaussians.count)
	);
	uint8_t* imageBufMem = createImageBuf(
		DDGSimageBuffers::required_mem<DDGSimageBuffers>(settings.width * settings.height)
	);
	
	DDGSgeomBuffers geomBufs = DDGS_CUDA_ERROR_CHECK(DDGSgeomBuffers(
		geomBufMem, gaussians.count
	));
	DDGSimageBuffers imageBufs = DDGS_CUDA_ERROR_CHECK(DDGSimageBuffers(
		imageBufMem, settings.width * settings.height
	));

	DDGS_PROFILE_REGION_END(allocateGeomImage);

	//preprocess:
	//---------------
	DDGS_PROFILE_REGION_START(preprocess);

	uint32_t numWorkgroupsPreprocess = _ddgs_ceildivide32(gaussians.count, DDGS_PREPROCESS_WORKGROUP_SIZE);
	_ddgs_foward_preprocess_kernel<<<numWorkgroupsPreprocess, DDGS_PREPROCESS_WORKGROUP_SIZE>>>(
		settings, gaussians, geomBufs
	);
	DDGS_CUDA_ERROR_CHECK();

	DDGS_PROFILE_REGION_END(preprocess);

	//prefix sum on tile counts:
	//---------------
	DDGS_PROFILE_REGION_START(tileCountScan);

	DDGS_CUDA_ERROR_CHECK(cub::DeviceScan::InclusiveSum(
		geomBufs.tilesTouchedScanTemp, geomBufs.tilesTouchedScanTempSize, geomBufs.tilesTouched, geomBufs.tilesTouchedScan, gaussians.count
	));

	uint32_t numRendered;
	DDGS_CUDA_ERROR_CHECK(cudaMemcpy(
		&numRendered, geomBufs.tilesTouchedScan + gaussians.count - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost
	));

	DDGS_PROFILE_REGION_END(tileCountScan);

	if(numRendered == 0)
		return 0;

	//allocate binning buffers:
	//---------------
	DDGS_PROFILE_REGION_START(allocateBinning);

	uint8_t* binningBufMem = createBinningBuf(
		DDGSrenderBuffers::required_mem<DDGSbinningBuffers>(numRendered)
	);

	DDGSbinningBuffers binningBufs = DDGS_CUDA_ERROR_CHECK(DDGSbinningBuffers(
		binningBufMem, numRendered
	));

	DDGS_PROFILE_REGION_END(allocateBinning);

	//write keys:
	//---------------
	DDGS_PROFILE_REGION_START(writeKeys);

	uint32_t numWorkgroupsWriteKeys = _ddgs_ceildivide32(gaussians.count, DDGS_KEY_WRITE_WORKGROUP_SIZE);
	_ddgs_forward_write_keys_kernel<<<numWorkgroupsWriteKeys, DDGS_KEY_WRITE_WORKGROUP_SIZE>>>(
		settings.width, settings.height,
		gaussians.count, geomBufs,
		binningBufs.keys, binningBufs.indices
	);
	DDGS_CUDA_ERROR_CHECK();

	DDGS_PROFILE_REGION_END(writeKeys);

	//sort keys:
	//---------------
	DDGS_PROFILE_REGION_START(sortKeys);

	uint32_t numTileBits = 0;
	while(tilesLen > 0)
	{
		numTileBits++;
		tilesLen >>= 1;
	}

	DDGS_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortPairs(
		binningBufs.sortTemp, binningBufs.sortTempSize,
		binningBufs.keys, binningBufs.keysSorted, binningBufs.indices, binningBufs.indicesSorted,
		numRendered, 0, 32 + numTileBits
	));

	DDGS_PROFILE_REGION_END(sortKeys);

	//get tile ranges:
	//---------------
	DDGS_PROFILE_REGION_START(tileRanges);

	DDGS_CUDA_ERROR_CHECK(cudaMemset(
		imageBufs.tileRanges, 0, tilesWidth * tilesHeight * sizeof(uint2)
	));

	uint32_t numWorkgroupsFindTileRanges = _ddgs_ceildivide32(numRendered, DDGS_FIND_TILE_RANGES_WORKGROUP_SIZE);
	_ddgs_forward_find_tile_ranges_kernel<<<numWorkgroupsFindTileRanges, DDGS_FIND_TILE_RANGES_WORKGROUP_SIZE>>>(
		numRendered, binningBufs.keysSorted, imageBufs.tileRanges
	);
	DDGS_CUDA_ERROR_CHECK();

	DDGS_PROFILE_REGION_END(tileRanges);

	//splat:
	//---------------
	DDGS_PROFILE_REGION_START(splat);

	_ddgs_forward_splat_kernel<<<{ tilesWidth, tilesHeight }, { DDGS_TILE_SIZE, DDGS_TILE_SIZE }>>>(
		settings,
		imageBufs.tileRanges, binningBufs.indicesSorted, geomBufs,
		outImg, outAlpha, imageBufs.transmittance, imageBufs.numContributors
	);
	DDGS_CUDA_ERROR_CHECK();

	DDGS_PROFILE_REGION_END(splat);

	//print timing information:
	//---------------
	DDGS_PROFILE_REGION_END(total);

#ifdef DDGS_PROFILE
	std::cout << std::endl;
	std::cout << "TOTAL FRAME TIME (forwards): " << DDGS_PROFILE_REGION_TIME(total) << "ms" << std::endl;
	std::cout << "\t- Allocating geom + image buffers: " << DDGS_PROFILE_REGION_TIME(allocateGeomImage) << "ms" << std::endl;
	std::cout << "\t- Preprocessing:                   " << DDGS_PROFILE_REGION_TIME(preprocess)        << "ms" << std::endl;
	std::cout << "\t- Tile count scan:                 " << DDGS_PROFILE_REGION_TIME(tileCountScan)     << "ms" << std::endl;
	std::cout << "\t- Allocating binning buffers:      " << DDGS_PROFILE_REGION_TIME(allocateBinning)   << "ms" << std::endl;
	std::cout << "\t- Writing render keys:             " << DDGS_PROFILE_REGION_TIME(writeKeys)         << "ms" << std::endl;
	std::cout << "\t- Sorting render keys:             " << DDGS_PROFILE_REGION_TIME(sortKeys)          << "ms" << std::endl;
	std::cout << "\t- Finding tile ranges:             " << DDGS_PROFILE_REGION_TIME(tileRanges)        << "ms" << std::endl;
	std::cout << "\t- Rasterizing:                     " << DDGS_PROFILE_REGION_TIME(splat)             << "ms" << std::endl;
	std::cout << std::endl;
#endif

	//return:
	//---------------
	return numRendered;
}

//-------------------------------------------//

__global__ static void __launch_bounds__(DDGS_PREPROCESS_WORKGROUP_SIZE)
_ddgs_foward_preprocess_kernel(DDGSsettings settings, DDGSgaussians gaussians, DDGSgeomBuffers outGeom)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= gaussians.count)
		return;

	outGeom.tilesTouched[idx] = 0; //so we dont render if culled
	outGeom.pixRadii[idx] = 0.0f;

	//find view and clip pos:
	//---------------
	QMvec3 mean = qm_vec3_load(&gaussians.means[idx * 3]);

	QMvec4 camPos = qm_mat4_mult_vec4(
		settings.view, 
		(QMvec4){ mean.x, mean.y, mean.z, 1.0f }
	);
	QMvec4 clipPos = qm_mat4_mult_vec4(
		settings.proj, camPos
	);

	//cull gaussians out of view:
	//---------------

	//TODO: tweak
	float clip = 1.2 * clipPos.w;
	if(clipPos.x >  clip || clipPos.y >  clip || clipPos.z >  clip ||
	   clipPos.x < -clip || clipPos.y < -clip || clipPos.z < -clip)
		return;

	//compute covariance matrix:
	//---------------
	QMmat4 scaleMat = qm_mat4_scale(qm_vec3_load(&gaussians.scales[idx * 3]));
	QMmat4 rotMat = qm_quaternion_to_mat4(qm_quaternion_load(&gaussians.rotations[idx * 4]));

	//TODO: add mat3 scale and rot functions to QM so we dont need to do a top_left
	QMmat3 M = qm_mat4_top_left(qm_mat4_mult(scaleMat, rotMat));
	QMmat3 cov = qm_mat3_mult(qm_mat3_transpose(M), M);

	//project covariance matrix to 2D:
	//---------------
	QMmat3 J = {{
		{ -settings.focalX / camPos.z, 0.0,                         (settings.focalX * camPos.x) / (camPos.z * camPos.z) },
		{ 0.0,                         -settings.focalY / camPos.z, (settings.focalY * camPos.y) / (camPos.z * camPos.z) },
		{ 0.0,                         0.0,                         0.0                                                  }
	}};

	QMmat3 W = qm_mat3_transpose(qm_mat4_top_left(settings.view));
	QMmat3 T = qm_mat3_mult(W, J);

	QMmat3 cov2d = qm_mat3_mult(
		qm_mat3_transpose(T),
		qm_mat3_mult(cov, T)
	);

	//compute inverse 2d covariance:
	//---------------
	float det = cov2d.m[0][0] * cov2d.m[1][1] - cov2d.m[0][1] * cov2d.m[0][1];
	if(det == 0.0f)
		return;

	QMvec3 conic = qm_vec3_scale((QMvec3){ cov2d.m[1][1], -cov2d.m[0][1], cov2d.m[0][0] }, 1.0f / det);

	//compute eigenvalues:
	//---------------
	float midpoint = (cov2d.m[0][0] + cov2d.m[1][1]) / 2.0;
	float radius = qm_vec2_length((QMvec2){ (cov2d.m[0][0] - cov2d.m[1][1]) / 2.0, cov2d.m[0][1] });

	float lambda1 = midpoint + radius;
	float lambda2 = midpoint - radius;

	//compute image tiles:
	//---------------
	QMvec2 pixCenter = {
		((clipPos.x / clipPos.w + 1.0f) * 0.5f * settings.width ) - 0.5f,
		((clipPos.y / clipPos.w + 1.0f) * 0.5f * settings.height) - 0.5f
	};

	//TODO: tweak
	float pixRadius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

	uint2 tilesMin, tilesMax;
	_ddgs_get_tile_bounds(
		settings.width, settings.height, pixCenter, pixRadius,
		tilesMin, tilesMax
	);

	if(tilesMin.x >= tilesMax.x || tilesMin.y >= tilesMax.y)
		return;

	//compute spherical harmonics:
	//---------------
	QMvec3 color = qm_vec3_load(&gaussians.harmonics[idx * 3]);

	//TODO: actual SH

	//write out:
	//---------------
	outGeom.pixCenters  [idx] = { pixCenter.x, pixCenter.y };
	outGeom.pixRadii    [idx] = pixRadius;
	outGeom.depths      [idx] = camPos.z;
	outGeom.tilesTouched[idx] = (tilesMax.x - tilesMin.x) * (tilesMax.y - tilesMin.y);
	outGeom.covs        [idx] = { cov.m[0][0], cov.m[1][0], cov.m[2][0], cov.m[1][1], cov.m[2][1], cov.m[2][2] };
	outGeom.conicOpacity[idx] = { conic.x, conic.y, conic.z, gaussians.opacities[idx] };
	outGeom.color       [idx] = { color.x, color.y, color.z };
}

__global__ static void __launch_bounds__(DDGS_KEY_WRITE_WORKGROUP_SIZE)
_ddgs_forward_write_keys_kernel(uint32_t width, uint32_t height,
                                uint32_t numGaussians, const DDGSgeomBuffers geom,
                                uint64_t* outKeys, uint32_t* outValues)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numGaussians)
		return;

	//skip if gaussian is not visible:
	//---------------
	if(geom.pixRadii[idx] == 0.0f)
		return;

	//get tile bounds:
	//---------------
	uint2 tilesMin, tilesMax;
	_ddgs_get_tile_bounds(
		width, height, geom.pixCenters[idx], geom.pixRadii[idx],
		tilesMin, tilesMax
	);

	//write keys:
	//---------------
	uint32_t writeIdx = (idx == 0) ? 0 : geom.tilesTouchedScan[idx - 1];
	uint32_t tilesWidth  = _ddgs_ceildivide32(width , DDGS_TILE_SIZE);

	for(uint32_t y = tilesMin.y; y < tilesMax.y; y++)
	for(uint32_t x = tilesMin.x; x < tilesMax.x; x++)
	{
		uint32_t tileIdx = x + tilesWidth * y;
		uint64_t key = ((uint64_t)tileIdx << 32) | *((uint32_t*)&geom.depths[idx]);

		outKeys[writeIdx] = key;
		outValues[writeIdx] = idx;
		writeIdx++;
	}
}

__global__ static void __launch_bounds__(DDGS_FIND_TILE_RANGES_WORKGROUP_SIZE)
_ddgs_forward_find_tile_ranges_kernel(uint32_t numRendered, const uint64_t* keys, uint2* outRanges)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numRendered)
		return;

	uint32_t tileIdx = (uint32_t)(keys[idx] >> 32);
	
	if(idx == 0)
		outRanges[tileIdx].x = 0;
	else
	{
		uint32_t prevTileIdx = (uint32_t)(keys[idx - 1] >> 32);
		if(tileIdx != prevTileIdx)
		{
			outRanges[prevTileIdx].y = idx;
			outRanges[tileIdx].x = idx;
		}
	}

	if(idx == numRendered - 1)
		outRanges[tileIdx].y = numRendered;
}

__global__ static void __launch_bounds__(DDGS_TILE_LEN)
_ddgs_forward_splat_kernel(DDGSsettings settings, 
                           const uint2* ranges, const uint32_t* indices, const DDGSgeomBuffers geom,
                           float* outColor, float* outAlpha, float* outTransmittance, uint32_t* outNumContributors)
{
	//compute pixel position:
	//---------------
	auto block = cg::this_thread_block();
	uint32_t tilesWidth = _ddgs_ceildivide32(settings.width, DDGS_TILE_SIZE);

	uint32_t pixelMinX = block.group_index().x * DDGS_TILE_SIZE;
	uint32_t pixelMinY = block.group_index().y * DDGS_TILE_SIZE;

	uint32_t pixelMaxX = min(pixelMinX + DDGS_TILE_SIZE, settings.width );
	uint32_t pixelMaxY = min(pixelMinY + DDGS_TILE_SIZE, settings.height);

	uint32_t pixelX = pixelMinX + block.thread_index().x;
	uint32_t pixelY = pixelMinY + block.thread_index().y;
	
	uint32_t pixelId = pixelX + settings.width * pixelY;

	bool inside = pixelX < settings.width && pixelY < settings.height;

	//read gaussian range:
	//---------------
	uint2 range = ranges[block.group_index().x + tilesWidth * block.group_index().y];
	int32_t numToRender = range.y - range.x;
	uint32_t numRounds = _ddgs_ceildivide32(numToRender, DDGS_TILE_LEN);

	//allocate shared memory:
	//---------------
	__shared__ QMvec2 collectedPixCenters  [DDGS_TILE_LEN];
	__shared__ QMvec4 collectedConicOpacity[DDGS_TILE_LEN];
	__shared__ QMvec3 collectedColor       [DDGS_TILE_LEN];

	//loop over batches until all threads are done:
	//---------------
	bool done = !inside;

	float accumTransmittance = 1.0f;
	QMvec3 accumColor = qm_vec3_full(0.0f);

	uint32_t numContributors = 0;
	uint32_t lastContributor = 0;

	for(uint32_t i = 0; i < numRounds; i++)
	{
		//exit early if all threads done
		int numDone = __syncthreads_count(done);
		if(numDone == DDGS_TILE_LEN)
			break;

		//collectively load gaussian data
		uint32_t loadIdx = i * DDGS_TILE_LEN + block.thread_rank();
		if(range.x + loadIdx < range.y)
		{
			uint32_t gaussianIdx = indices[range.x + loadIdx];
			collectedPixCenters  [block.thread_rank()] = geom.pixCenters[gaussianIdx];
			collectedConicOpacity[block.thread_rank()] = geom.conicOpacity[gaussianIdx];
			collectedColor       [block.thread_rank()] = geom.color[gaussianIdx];
		}

		block.sync();

		//accumulate collected gaussians
		for(uint32_t j = 0; j < min(DDGS_TILE_LEN, numToRender); j++)
		{
			numContributors++;

			QMvec2 pos = collectedPixCenters[j];
			QMvec4 conicO = collectedConicOpacity[j];

			float dx = pos.x - (float)pixelX;
			float dy = pos.y - (float)pixelY;

			float power = -0.5f * (conicO.x * dx * dx + conicO.z * dy * dy) - conicO.y * dx * dy;
			if(power > 0.0f)
				continue;

			float alpha = min(DDGS_MAX_ALPHA, conicO.w * exp(power));
			if(alpha < DDGS_MIN_ALPHA)
				continue;
			
			float newAccumTramsittance = accumTransmittance * (1.0f - alpha);
			if(newAccumTramsittance < DDGS_TRANSMITTANCE_CUTOFF)
			{
				done = true;
				continue;
			}

			accumColor = qm_vec3_add(accumColor, qm_vec3_scale(collectedColor[j], alpha * accumTransmittance));
			accumTransmittance = newAccumTramsittance;

			lastContributor = numContributors;
		}

		//decrement num left to render
		numToRender -= DDGS_TILE_LEN;
	}

	//write final color:
	//---------------
	if(inside)
	{
		outColor[pixelId * 3 + 0] = accumColor.r;
		outColor[pixelId * 3 + 1] = accumColor.g;
		outColor[pixelId * 3 + 2] = accumColor.b;
		outAlpha[pixelId] = 1.0f - accumTransmittance;

		outTransmittance[pixelId] = accumTransmittance;
		outNumContributors[pixelId] = lastContributor;
	}
}

__device__ static void _ddgs_get_tile_bounds(uint32_t width, uint32_t height, QMvec2 pixCenter, float pixRadius, uint2& tileMin, uint2& tileMax)
{
	//TODO: use a smarter method to generate tiles
	//this generates far too many tiles for very anisotropic gaussians

	uint32_t tilesWidth  = _ddgs_ceildivide32(width , DDGS_TILE_SIZE);
	uint32_t tilesHeight = _ddgs_ceildivide32(height, DDGS_TILE_SIZE);
 
	tileMin.x = (uint32_t)min(max((int32_t)((pixCenter.x - pixRadius)                        / DDGS_TILE_SIZE), 0), tilesWidth );
	tileMin.y = (uint32_t)min(max((int32_t)((pixCenter.y - pixRadius)                        / DDGS_TILE_SIZE), 0), tilesHeight);
	tileMax.x = (uint32_t)min(max((int32_t)((pixCenter.x + pixRadius + DDGS_TILE_SIZE - 1) / DDGS_TILE_SIZE), 0), tilesWidth );
	tileMax.y = (uint32_t)min(max((int32_t)((pixCenter.y + pixRadius + DDGS_TILE_SIZE - 1) / DDGS_TILE_SIZE), 0), tilesHeight);
}