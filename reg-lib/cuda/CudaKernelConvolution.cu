#include "CudaKernelConvolution.hpp"

/* *************************************************************** */
void NiftyReg::Cuda::KernelConvolutionWorkspace::EnsureSize(const size_t voxelNumber, const int channels) {
    const size_t intensitySize = voxelNumber * channels;
    if (intensityA.size() < intensitySize) {
        intensityA.resize(intensitySize);
        intensityB.resize(intensitySize);
    }
    if (density.size() < voxelNumber) {
        density.resize(voxelNumber);
        densityAux.resize(voxelNumber);
        nanImage.resize(voxelNumber);
    }
}
/* *************************************************************** */
namespace {
/* *************************************************************** */
using NiftyReg::Cuda::KernelConvolutionWorkspace;
/* *************************************************************** */
// L2 cache size of the current device (queried once); used to decide when an axis pass's sliding
// window would overflow L2 and the column-parallel variant should be used instead.
size_t GetL2CacheSize() {
    static const size_t l2CacheSize = []() {
        int device = 0, l2Size = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, device);
        return l2Size > 0 ? static_cast<size_t>(l2Size) : size_t(4) << 20;
    }();
    return l2CacheSize;
}
/* *************************************************************** */
// Fill the kernel weights on the host (identical maths to the CPU reg_tools_kernelConvolution)
// and return the kernel sum. Shared by the single-channel and packed convolutions.
template<ConvKernelType kernelType>
double ComputeKernelWeights(vector<float>& kernel, const int radius, const double temp) {
    kernel.resize(2 * radius + 1);
    double kernelSum = 0;
    if constexpr (kernelType == ConvKernelType::Cubic) {
        for (int i = -radius; i <= radius; i++) {
            double relative = fabs(i / temp);
            if (relative < 1.0)
                kernel[i + radius] = static_cast<float>(2.0 / 3.0 - Square(relative) + 0.5 * Cube(relative));
            else if (relative < 2.0)
                kernel[i + radius] = static_cast<float>(-Cube(relative - 2.0) / 6.0);
            else kernel[i + radius] = 0;
            kernelSum += kernel[i + radius];
        }
    } else if constexpr (kernelType == ConvKernelType::Gaussian) {
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = static_cast<float>(exp(-Square(i) / (2.0 * Square(temp))) / (temp * 2.506628274631));
            kernelSum += kernel[i + radius];
        }
    } else if constexpr (kernelType == ConvKernelType::Linear) {
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = 1.f - fabs(i / static_cast<float>(radius));
            kernelSum += kernel[i + radius];
        }
    } else if constexpr (kernelType == ConvKernelType::Mean) {
        // Only reached in 2D (3D Mean uses the cumulative path)
        for (int i = -radius; i <= radius; i++) {
            kernel[i + radius] = 1.f;
            kernelSum += kernel[i + radius];
        }
    }
    return kernelSum;
}
/* *************************************************************** */
// Upload the kernel weights and return the device pointer. When a workspace is available the
// upload is skipped if the same kernel (type, radius, spacing) is already resident - consecutive
// convolutions of one LNCC call all share the same kernel, so only the first axis pays the copy.
template<ConvKernelType kernelType>
const float* UploadKernelWeights(const vector<float>& kernel, const int radius, const double temp,
                                 KernelConvolutionWorkspace *workspace,
                                 thrust::device_vector<float>& localKernel) {
    if (workspace) {
        if (workspace->cachedKernelKind != int(kernelType) ||
            workspace->cachedKernelRadius != radius ||
            workspace->cachedKernelTemp != temp) {
            if (workspace->kernel.size() < kernel.size()) workspace->kernel.resize(kernel.size());
            thrust::copy(kernel.begin(), kernel.end(), workspace->kernel.begin());
            workspace->cachedKernelKind = int(kernelType);
            workspace->cachedKernelRadius = radius;
            workspace->cachedKernelTemp = temp;
        }
        return workspace->kernel.data().get();
    }
    localKernel = kernel;
    return localKernel.data().get();
}
/* *************************************************************** */
} // anonymous namespace
/* *************************************************************** */
template<ConvKernelType kernelType, class AccType>
void NiftyReg::Cuda::KernelConvolution(const nifti_image *image,
                                       float4 *imageCuda,
                                       const float *sigma,
                                       const bool *timePoints,
                                       const bool *axes,
                                       const int *maskCuda,
                                       KernelConvolutionWorkspace *workspace) {
    if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048)
        NR_FATAL_ERROR("This function does not support images with dimensions larger than 2048");

    bool axesToSmooth[3];
    if (axes == nullptr) {
        // All axes are smoothed by default
        axesToSmooth[0] = axesToSmooth[1] = axesToSmooth[2] = true;
    } else for (int i = 0; i < 3; i++) axesToSmooth[i] = axes[i];

    const auto activeTimePointCount = std::min(image->nt * image->nu, 4);
    bool activeTimePoints[4]{}; // 4 is the maximum number of time points
    if (timePoints == nullptr) {
        // All time points are considered as active
        for (auto i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = true;
    } else for (auto i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = timePoints[i];

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDims = make_int3(image->nx, image->ny, image->nz);

    // Scratch buffers: reuse the caller's workspace when provided (avoids per-call cudaMalloc/
    // cudaFree), else allocate locally. `computeDensity` is false only when the caller has a valid
    // cached density to reuse (see KernelConvolutionWorkspace), in which case the density is neither
    // recomputed nor convolved and the cached smoothed density is used for the final normalisation.
    thrust::device_vector<float> localDensity, localDensityAux, localIntensityA, localIntensityB;
    thrust::device_vector<bool> localNanImage;
    float *densityPtr, *densityAuxPtr, *intensityAPtr, *intensityBPtr;
    bool *nanImagePtr;
    bool computeDensity = true;
    if (workspace) {
        workspace->EnsureSize(voxelNumber);
        densityPtr = workspace->density.data().get();
        densityAuxPtr = workspace->densityAux.data().get();
        nanImagePtr = workspace->nanImage.data().get();
        intensityAPtr = workspace->intensityA.data().get();
        intensityBPtr = workspace->intensityB.data().get();
        computeDensity = !workspace->densityValid;
    } else {
        localDensity.resize(voxelNumber);
        localDensityAux.resize(voxelNumber);
        localNanImage.resize(voxelNumber);
        localIntensityA.resize(voxelNumber);
        localIntensityB.resize(voxelNumber);
        densityPtr = localDensity.data().get();
        densityAuxPtr = localDensityAux.data().get();
        nanImagePtr = localNanImage.data().get();
        intensityAPtr = localIntensityA.data().get();
        intensityBPtr = localIntensityB.data().get();
    }
    // Where a cached density (computeDensity == false) is to be read from: the weighted path may
    // have left it in either buffer (see KernelConvolutionWorkspace::densityInAux).
    float *cachedDensityPtr = workspace && workspace->densityInAux ? densityAuxPtr : densityPtr;

    // The cumulative (running-sum) filter is only used for the Mean kernel on 3D volumes; every
    // other case (all non-Mean kernels, and the Mean kernel in 2D) uses a weighted sum. A single
    // convolution never mixes the two, so the whole call takes one branch.
    constexpr bool isMean = kernelType == ConvKernelType::Mean;
    const bool cumulative = isMean && imageDims.z > 1;

    for (int t = 0; t < activeTimePointCount; t++) {
        if (!activeTimePoints[t]) continue;

        if (cumulative) {
            // Cumulative (3D Mean) path: line-parallel and in-place, unchanged from the original -
            // the running-sum filter cannot be voxel-parallelised bit-exactly. The in-place filter
            // works in the stable density buffer; a cached density is only read (final normalisation).
            float *activeDensityPtr = computeDensity ? densityPtr : cachedDensityPtr;
            auto imageTexturePtr = Cuda::CreateTextureObject(imageCuda, voxelNumber, cudaChannelFormatKindFloat, 1);
            auto densityTexturePtr = Cuda::CreateTextureObject(activeDensityPtr, voxelNumber, cudaChannelFormatKindFloat, 1);
            auto imageTexture = *imageTexturePtr;
            auto densityTexture = *densityTexturePtr;
            // Reuse intensityA/intensityB as the per-plane intensity/density line buffers
            float *bufferIntensityCudaPtr = intensityAPtr;
            float *bufferDensityCudaPtr = intensityBPtr;

            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
                if (computeDensity) {
                    const float intensityVal = tex1Dfetch<float>(imageTexture, index * 4 + t);
                    const bool inMask = maskCuda == nullptr || maskCuda[index] >= 0;
                    const float d = inMask && intensityVal == intensityVal ? 1.f : 0;
                    activeDensityPtr[index] = d;
                    nanImagePtr[index] = !static_cast<bool>(d);
                    if (!static_cast<bool>(d)) reinterpret_cast<float*>(&imageCuda[index])[t] = 0;
                } else {
                    if (nanImagePtr[index]) reinterpret_cast<float*>(&imageCuda[index])[t] = 0;
                }
            });

            for (int n = 0; n < 3; n++) {
                if (!axesToSmooth[n] || image->dim[n] <= 1) continue;
                double temp;
                if (sigma[t] > 0) temp = sigma[t] / image->pixdim[n + 1];
                else temp = fabs(sigma[t]);
                const int radius = static_cast<int>(temp);
                if (radius <= 0) continue;

                int planeCount, lineOffset;
                switch (n) {
                case 0: planeCount = imageDims.y * imageDims.z; lineOffset = 1; break;
                case 1: planeCount = imageDims.x * imageDims.z; lineOffset = imageDims.x; break;
                case 2: planeCount = imageDims.x * imageDims.y; lineOffset = planeCount; break;
                }
                const int imageDim = reinterpret_cast<const int*>(&imageDims)[n];

                thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), planeCount, [=]__device__(const int planeIndex) {
                    int realIndex = 0;
                    switch (n) {
                    case 0: realIndex = planeIndex * imageDims.x; break;
                    case 1: realIndex = (planeIndex / imageDims.x) * imageDims.x * imageDims.y + planeIndex % imageDims.x; break;
                    case 2: realIndex = planeIndex; break;
                    }
                    const auto bufferIndex = planeIndex * imageDim;
                    float *bufferIntensityPtr = &bufferIntensityCudaPtr[bufferIndex];
                    float *bufferDensityPtr = &bufferDensityCudaPtr[bufferIndex];
                    for (int lineIndex = 0, index = realIndex; lineIndex < imageDim; lineIndex++, index += lineOffset) {
                        bufferIntensityPtr[lineIndex] = tex1Dfetch<float>(imageTexture, index * 4 + t);
                        if (computeDensity) bufferDensityPtr[lineIndex] = tex1Dfetch<float>(densityTexture, index);
                    }
                    for (int lineIndex = 1; lineIndex < imageDim; lineIndex++) {
                        bufferIntensityPtr[lineIndex] += bufferIntensityPtr[lineIndex - 1];
                        if (computeDensity) bufferDensityPtr[lineIndex] += bufferDensityPtr[lineIndex - 1];
                    }
                    int shiftPre = -radius - 1;
                    int shiftPst = radius;
                    for (int lineIndex = 0; lineIndex < imageDim; lineIndex++, shiftPre++, shiftPst++, realIndex += lineOffset) {
                        float bufferIntensityCur, bufferDensityCur = 0;
                        if (shiftPre > -1) {
                            if (shiftPst < imageDim) {
                                bufferIntensityCur = bufferIntensityPtr[shiftPre] - bufferIntensityPtr[shiftPst];
                                if (computeDensity) bufferDensityCur = bufferDensityPtr[shiftPre] - bufferDensityPtr[shiftPst];
                            } else {
                                bufferIntensityCur = bufferIntensityPtr[shiftPre] - bufferIntensityPtr[imageDim - 1];
                                if (computeDensity) bufferDensityCur = bufferDensityPtr[shiftPre] - bufferDensityPtr[imageDim - 1];
                            }
                        } else {
                            if (shiftPst < imageDim) {
                                bufferIntensityCur = -bufferIntensityPtr[shiftPst];
                                if (computeDensity) bufferDensityCur = -bufferDensityPtr[shiftPst];
                            } else {
                                bufferIntensityCur = 0;
                                bufferDensityCur = 0;
                            }
                        }
                        reinterpret_cast<float*>(&imageCuda[realIndex])[t] = bufferIntensityCur;
                        if (computeDensity) activeDensityPtr[realIndex] = bufferDensityCur;
                    }
                });
            } // axes

            if (workspace && computeDensity) workspace->densityInAux = false;

            // Normalise (or NaN out-of-mask)
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
                float *intensityPtr = &reinterpret_cast<float*>(&imageCuda[index])[t];
                if (nanImagePtr[index]) *intensityPtr = std::numeric_limits<float>::quiet_NaN();
                else *intensityPtr = *intensityPtr / activeDensityPtr[index];
            });
        } else {
            /* Weighted (voxel-parallel) path: one thread per output voxel, reading its
             * (2*radius+1)-tap neighbourhood directly via __ldg (overlapping reads hit the read-only
             * cache). Intensity ping-pongs between intensityA/intensityB and density between
             * density/densityAux; source != destination each axis, so no read-after-write hazard. */
            float *curIntensity = intensityAPtr, *nextIntensity = intensityBPtr;
            float *curDensity = computeDensity ? densityPtr : cachedDensityPtr;
            float *nextDensity = curDensity == densityPtr ? densityAuxPtr : densityPtr;
            {
                float *initDensity = curDensity;
                thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
                    const float intensityVal = reinterpret_cast<float*>(&imageCuda[index])[t];
                    if (computeDensity) {
                        const bool inMask = maskCuda == nullptr || maskCuda[index] >= 0;
                        const bool active = inMask && intensityVal == intensityVal;
                        initDensity[index] = active ? 1.f : 0;
                        nanImagePtr[index] = !active;
                        curIntensity[index] = active ? intensityVal : 0;
                    } else {
                        curIntensity[index] = nanImagePtr[index] ? 0 : intensityVal;
                    }
                });
            }

            for (int n = 0; n < 3; n++) {
                if (!axesToSmooth[n] || image->dim[n] <= 1) continue;

                double temp;
                if (sigma[t] > 0) temp = sigma[t] / image->pixdim[n + 1]; // mm to voxel
                else temp = fabs(sigma[t]); // voxel-based if negative value
                int radius = 0;
                if constexpr (kernelType == ConvKernelType::Mean || kernelType == ConvKernelType::Linear)
                    radius = static_cast<int>(temp);
                else if constexpr (kernelType == ConvKernelType::Gaussian)
                    radius = static_cast<int>(temp * 3.0);
                else if constexpr (kernelType == ConvKernelType::Cubic)
                    radius = static_cast<int>(temp * 2.0);
                else NR_FATAL_ERROR("Unknown kernel type");
                if (radius <= 0) continue;

                // Fill the kernel (identical maths to the CPU reg_tools_kernelConvolution)
                vector<float> kernel;
                const double kernelSum = ComputeKernelWeights<kernelType>(kernel, radius, temp);
                NR_DEBUG("Convolution type[" << int(kernelType) << "] dim[" << n << "] tp[" << t << "] radius[" << radius << "] kernelSum[" << kernelSum << "]");
                if (kernelSum <= 0) continue; // no weighting to apply

                // Upload the kernel weights (cached in the workspace when available)
                thrust::device_vector<float> localKernel;
                const float *kernelPtr = UploadKernelWeights<kernelType>(kernel, radius, temp, workspace, localKernel);

                const int nx = imageDims.x, ny = imageDims.y;
                const int imageDim = reinterpret_cast<const int*>(&imageDims)[n];
                const size_t stride = n == 0 ? 1 : (n == 1 ? (size_t)nx : (size_t)nx * ny);
                const float *srcIntensity = curIntensity, *srcDensity = curDensity;
                float *dstIntensity = nextIntensity, *dstDensity = nextDensity;
                const bool doDensity = computeDensity;

                // See the packed variant: when the z pass's (2*radius+1)-plane sliding window
                // overflows L2, march one thread per (x,y) column along z instead of one thread per
                // voxel - identical per-voxel tap order, so bit-for-bit the same result.
                const size_t slidingWindowBytes = static_cast<size_t>(2 * radius + 1) * nx * ny
                    * (doDensity ? 2 : 1) * sizeof(float);
                const bool columnParallel = n == 2 && slidingWindowBytes > GetL2CacheSize() / 2;

                if (columnParallel) {
                    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), (size_t)nx * ny, [=]__device__(const size_t lineStart) {
                        for (int linePos = 0; linePos < imageDim; ++linePos) {
                            const size_t index = lineStart + static_cast<size_t>(linePos) * stride;
                            int shiftPre = linePos - radius;
                            int shiftPst = linePos + radius + 1;
                            int kernelIndex = 0;
                            if (shiftPre < 0) { kernelIndex = -shiftPre; shiftPre = 0; }
                            if (shiftPst > imageDim) shiftPst = imageDim;
                            AccType intensitySum = 0, densitySum = 0;
                            for (int k = shiftPre; k < shiftPst; k++, kernelIndex++) {
                                const float kernelValue = kernelPtr[kernelIndex];
                                const size_t idx = lineStart + static_cast<size_t>(k) * stride;
                                intensitySum += kernelValue * __ldg(&srcIntensity[idx]);
                                if (doDensity) densitySum += kernelValue * __ldg(&srcDensity[idx]);
                            }
                            dstIntensity[index] = static_cast<float>(intensitySum);
                            if (doDensity) dstDensity[index] = static_cast<float>(densitySum);
                        }
                    });
                } else
                thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
                    // Position of this voxel along the current axis and the start of its line
                    int linePos;
                    size_t lineStart;
                    if (n == 0) {
                        linePos = static_cast<int>(index % nx);
                        lineStart = index - linePos;
                    } else if (n == 1) {
                        const int x = static_cast<int>(index % nx);
                        const size_t rem = index / nx;
                        linePos = static_cast<int>(rem % ny);
                        const int z = static_cast<int>(rem / ny);
                        lineStart = static_cast<size_t>(x) + static_cast<size_t>(z) * nx * ny;
                    } else {
                        const size_t plane = (size_t)nx * ny;
                        linePos = static_cast<int>(index / plane);
                        lineStart = index % plane;
                    }
                    int shiftPre = linePos - radius;
                    int shiftPst = linePos + radius + 1;
                    int kernelIndex = 0;
                    if (shiftPre < 0) { kernelIndex = -shiftPre; shiftPre = 0; }
                    if (shiftPst > imageDim) shiftPst = imageDim;
                    // AccType (double by default, float for LNCC) - matches the CPU accumulation
                    AccType intensitySum = 0, densitySum = 0;
                    for (int k = shiftPre; k < shiftPst; k++, kernelIndex++) {
                        const float kernelValue = kernelPtr[kernelIndex];
                        const size_t idx = lineStart + static_cast<size_t>(k) * stride;
                        intensitySum += kernelValue * __ldg(&srcIntensity[idx]);
                        if (doDensity) densitySum += kernelValue * __ldg(&srcDensity[idx]);
                    }
                    dstIntensity[index] = static_cast<float>(intensitySum);
                    if (doDensity) dstDensity[index] = static_cast<float>(densitySum);
                });

                std::swap(curIntensity, nextIntensity);
                if (computeDensity) std::swap(curDensity, nextDensity);
            } // axes

            // Record which buffer holds the smoothed density (avoids a whole-volume copy; the
            // location is honoured by subsequent density-reusing convolutions via densityInAux)
            if (workspace && computeDensity) workspace->densityInAux = curDensity == densityAuxPtr;

            // Normalise (or NaN out-of-mask)
            const float *finalIntensity = curIntensity;
            const float *finalDensity = curDensity;
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
                float *intensityPtr = &reinterpret_cast<float*>(&imageCuda[index])[t];
                if (nanImagePtr[index]) *intensityPtr = std::numeric_limits<float>::quiet_NaN();
                else *intensityPtr = finalIntensity[index] / finalDensity[index];
            });
        } // weighted path
    } // check if the time point is active

    // The workspace now holds the smoothed density and NaN mask for this call's mask; a subsequent
    // convolution sharing the same density can reuse them (the caller resets densityValid when the
    // mask/density changes).
    if (workspace)
        workspace->densityValid = true;
}
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Mean, double>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Linear, double>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Gaussian, double>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Cubic, double>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
// float-accumulation variants (used by LNCC for a faster, still CPU-bit-exact convolution)
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Mean, float>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Linear, float>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Gaussian, float>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolution<ConvKernelType::Cubic, float>(const nifti_image*, float4*, const float*, const bool*, const bool*, const int*, KernelConvolutionWorkspace*);
/* *************************************************************** */
template<ConvKernelType kernelType, class AccType>
void NiftyReg::Cuda::KernelConvolutionPacked(const nifti_image *image,
                                             float4 *imageCuda,
                                             const float *sigma,
                                             const int *maskCuda,
                                             KernelConvolutionWorkspace *workspace) {
    if (!workspace)
        NR_FATAL_ERROR("KernelConvolutionPacked requires a workspace");
    if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048)
        NR_FATAL_ERROR("This function does not support images with dimensions larger than 2048");

    constexpr bool isMean = kernelType == ConvKernelType::Mean;
    if (isMean && image->nz > 1) {
        // The cumulative (running-sum) 3D-Mean filter cannot be voxel-parallelised bit-exactly, so
        // fall back to the single-channel path applied per channel. Presenting a four-time-point
        // header makes it walk the four float4 components, each with the same sigma.
        nifti_image header = *image; // shallow copy: the convolution only reads the geometry fields
        header.nt = header.dim[4] = 4;
        header.nu = header.dim[5] = 1;
        const float sigma4[4]{ sigma[0], sigma[0], sigma[0], sigma[0] };
        KernelConvolution<kernelType, AccType>(&header, imageCuda, sigma4, nullptr, nullptr, maskCuda, workspace);
        return;
    }

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDims = make_int3(image->nx, image->ny, image->nz);

    workspace->EnsureSize(voxelNumber, 4);
    float4 *intensityAPtr = reinterpret_cast<float4*>(workspace->intensityA.data().get());
    float4 *intensityBPtr = reinterpret_cast<float4*>(workspace->intensityB.data().get());
    float *densityPtr = workspace->density.data().get();
    float *densityAuxPtr = workspace->densityAux.data().get();
    bool *nanImagePtr = workspace->nanImage.data().get();
    const bool computeDensity = !workspace->densityValid;

    // The four channels share one density (see the header contract: identical mask/NaN pattern
    // across channels, with the density derived from lane .x), so the density work is exactly that
    // of a single-channel convolution and the cached density remains reusable either way.
    float4 *curIntensity = intensityAPtr, *nextIntensity = intensityBPtr;
    float *curDensity = computeDensity ? densityPtr
                                       : (workspace->densityInAux ? densityAuxPtr : densityPtr);
    float *nextDensity = curDensity == densityPtr ? densityAuxPtr : densityPtr;

    // Initialise the working buffer: zero out-of-density voxels (all lanes - the NaN pattern is
    // shared across channels) and, when required, compute the density and NaN mask from lane .x.
    {
        float *initDensity = curDensity;
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
            const float4 val = imageCuda[index];
            if (computeDensity) {
                const bool inMask = maskCuda == nullptr || maskCuda[index] >= 0;
                const bool active = inMask && val.x == val.x;
                initDensity[index] = active ? 1.f : 0;
                nanImagePtr[index] = !active;
                curIntensity[index] = active ? val : make_float4(0, 0, 0, 0);
            } else {
                curIntensity[index] = nanImagePtr[index] ? make_float4(0, 0, 0, 0) : val;
            }
        });
    }

    for (int n = 0; n < 3; n++) {
        if (image->dim[n] <= 1) continue;

        double temp;
        if (sigma[0] > 0) temp = sigma[0] / image->pixdim[n + 1]; // mm to voxel
        else temp = fabs(sigma[0]); // voxel-based if negative value
        int radius = 0;
        if constexpr (kernelType == ConvKernelType::Mean || kernelType == ConvKernelType::Linear)
            radius = static_cast<int>(temp);
        else if constexpr (kernelType == ConvKernelType::Gaussian)
            radius = static_cast<int>(temp * 3.0);
        else if constexpr (kernelType == ConvKernelType::Cubic)
            radius = static_cast<int>(temp * 2.0);
        else NR_FATAL_ERROR("Unknown kernel type");
        if (radius <= 0) continue;

        vector<float> kernel;
        const double kernelSum = ComputeKernelWeights<kernelType>(kernel, radius, temp);
        NR_DEBUG("Packed convolution type[" << int(kernelType) << "] dim[" << n << "] radius[" << radius << "] kernelSum[" << kernelSum << "]");
        if (kernelSum <= 0) continue; // no weighting to apply

        thrust::device_vector<float> localKernel;
        const float *kernelPtr = UploadKernelWeights<kernelType>(kernel, radius, temp, workspace, localKernel);

        const int nx = imageDims.x, ny = imageDims.y;
        const int imageDim = reinterpret_cast<const int*>(&imageDims)[n];
        const size_t stride = n == 0 ? 1 : (n == 1 ? (size_t)nx : (size_t)nx * ny);
        const float4 *srcIntensity = curIntensity;
        const float *srcDensity = curDensity;
        float4 *dstIntensity = nextIntensity;
        float *dstDensity = nextDensity;
        const bool doDensity = computeDensity;

        // The z pass reads taps a whole plane apart: in voxel-parallel order its cache working set
        // is (2*radius+1) planes of float4, which overflows L2 on large volumes and makes every tap
        // a DRAM access (measured ~17x slower at 256^3). When that window does not fit, march one
        // thread per (x,y) column along z instead: each thread then re-reads only its own taps,
        // whose live set (a block's columns x kernel width) stays cache-resident. The per-voxel tap
        // order is identical in both variants, so the results are bit-for-bit the same either way.
        const size_t slidingWindowBytes = static_cast<size_t>(2 * radius + 1) * nx * ny
            * (sizeof(float4) + (doDensity ? sizeof(float) : 0));
        const bool columnParallel = n == 2 && slidingWindowBytes > GetL2CacheSize() / 2;

        if (columnParallel) {
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), (size_t)nx * ny, [=]__device__(const size_t lineStart) {
                for (int linePos = 0; linePos < imageDim; ++linePos) {
                    const size_t index = lineStart + static_cast<size_t>(linePos) * stride;
                    int shiftPre = linePos - radius;
                    int shiftPst = linePos + radius + 1;
                    int kernelIndex = 0;
                    if (shiftPre < 0) { kernelIndex = -shiftPre; shiftPre = 0; }
                    if (shiftPst > imageDim) shiftPst = imageDim;
                    AccType sumX = 0, sumY = 0, sumZ = 0, sumW = 0, densitySum = 0;
                    for (int k = shiftPre; k < shiftPst; k++, kernelIndex++) {
                        const float kernelValue = kernelPtr[kernelIndex];
                        const size_t idx = lineStart + static_cast<size_t>(k) * stride;
                        const float4 val = __ldg(&srcIntensity[idx]);
                        sumX += kernelValue * val.x;
                        sumY += kernelValue * val.y;
                        sumZ += kernelValue * val.z;
                        sumW += kernelValue * val.w;
                        if (doDensity) densitySum += kernelValue * __ldg(&srcDensity[idx]);
                    }
                    dstIntensity[index] = make_float4(static_cast<float>(sumX), static_cast<float>(sumY),
                                                      static_cast<float>(sumZ), static_cast<float>(sumW));
                    if (doDensity) dstDensity[index] = static_cast<float>(densitySum);
                }
            });
        } else
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
            // Position of this voxel along the current axis and the start of its line
            int linePos;
            size_t lineStart;
            if (n == 0) {
                linePos = static_cast<int>(index % nx);
                lineStart = index - linePos;
            } else if (n == 1) {
                const int x = static_cast<int>(index % nx);
                const size_t rem = index / nx;
                linePos = static_cast<int>(rem % ny);
                const int z = static_cast<int>(rem / ny);
                lineStart = static_cast<size_t>(x) + static_cast<size_t>(z) * nx * ny;
            } else {
                const size_t plane = (size_t)nx * ny;
                linePos = static_cast<int>(index / plane);
                lineStart = index % plane;
            }
            int shiftPre = linePos - radius;
            int shiftPst = linePos + radius + 1;
            int kernelIndex = 0;
            if (shiftPre < 0) { kernelIndex = -shiftPre; shiftPre = 0; }
            if (shiftPst > imageDim) shiftPst = imageDim;
            // Four independent accumulations, each in the same tap order as a single-channel
            // convolution of that lane - hence bit-exact per lane
            AccType sumX = 0, sumY = 0, sumZ = 0, sumW = 0, densitySum = 0;
            for (int k = shiftPre; k < shiftPst; k++, kernelIndex++) {
                const float kernelValue = kernelPtr[kernelIndex];
                const size_t idx = lineStart + static_cast<size_t>(k) * stride;
                const float4 val = __ldg(&srcIntensity[idx]);
                sumX += kernelValue * val.x;
                sumY += kernelValue * val.y;
                sumZ += kernelValue * val.z;
                sumW += kernelValue * val.w;
                if (doDensity) densitySum += kernelValue * __ldg(&srcDensity[idx]);
            }
            dstIntensity[index] = make_float4(static_cast<float>(sumX), static_cast<float>(sumY),
                                              static_cast<float>(sumZ), static_cast<float>(sumW));
            if (doDensity) dstDensity[index] = static_cast<float>(densitySum);
        });

        std::swap(curIntensity, nextIntensity);
        if (computeDensity) std::swap(curDensity, nextDensity);
    } // axes

    // Record which buffer holds the smoothed density (no copy needed; readers follow densityInAux)
    if (computeDensity) workspace->densityInAux = curDensity == densityAuxPtr;

    // Normalise (or NaN out-of-mask)
    const float4 *finalIntensity = curIntensity;
    const float *finalDensity = curDensity;
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
        if (nanImagePtr[index]) {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            imageCuda[index] = make_float4(nan, nan, nan, nan);
        } else {
            const float4 val = finalIntensity[index];
            const float density = finalDensity[index];
            imageCuda[index] = make_float4(val.x / density, val.y / density,
                                           val.z / density, val.w / density);
        }
    });

    workspace->densityValid = true;
}
// Only the float-accumulation variants are instantiated: LNCC is the sole caller (see the header
// contract) and uses float accumulation throughout.
template void NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Mean, float>(const nifti_image*, float4*, const float*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Linear, float>(const nifti_image*, float4*, const float*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Gaussian, float>(const nifti_image*, float4*, const float*, const int*, KernelConvolutionWorkspace*);
template void NiftyReg::Cuda::KernelConvolutionPacked<ConvKernelType::Cubic, float>(const nifti_image*, float4*, const float*, const int*, KernelConvolutionWorkspace*);
/* *************************************************************** */
