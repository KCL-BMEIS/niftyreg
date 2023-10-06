#include "CudaKernelConvolution.hpp"

/* *************************************************************** */
void NiftyReg::Cuda::KernelConvolution(const nifti_image *image,
                                       float4 *imageCuda,
                                       const float *sigma,
                                       const int kernelType,
                                       const bool *timePoints,
                                       const bool *axis) {
    if (image->nx > 2048 || image->ny > 2048 || image->nz > 2048)
        NR_FATAL_ERROR("This function does not support images with dimensions larger than 2048");

    bool axisToSmooth[3];
    if (axis == nullptr) {
        // All axis are smoothed by default
        axisToSmooth[0] = axisToSmooth[1] = axisToSmooth[2] = true;
    } else for (int i = 0; i < 3; i++) axisToSmooth[i] = axis[i];

    const auto activeTimePointCount = std::min(image->nt * image->nu, 4);
    bool activeTimePoints[4]{}; // 4 is the maximum number of time points
    if (timePoints == nullptr) {
        // All time points are considered as active
        for (auto i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = true;
    } else for (auto i = 0; i < activeTimePointCount; i++) activeTimePoints[i] = timePoints[i];

    const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
    const int3 imageDims = make_int3(image->nx, image->ny, image->nz);

    thrust::device_vector<float> densityCuda(voxelNumber);
    thrust::device_vector<bool> nanImageCuda(voxelNumber);
    thrust::device_vector<float> bufferIntensityCuda(voxelNumber);
    thrust::device_vector<float> bufferDensityCuda(voxelNumber);
    float *densityCudaPtr = densityCuda.data().get();
    bool *nanImageCudaPtr = nanImageCuda.data().get();
    float *bufferIntensityCudaPtr = bufferIntensityCuda.data().get();
    float *bufferDensityCudaPtr = bufferDensityCuda.data().get();

    for (int t = 0; t < activeTimePointCount; t++) {
        if (!activeTimePoints[t]) continue;

        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
            float& intensityVal = reinterpret_cast<float*>(&imageCuda[index])[t];
            float& densityVal = densityCudaPtr[index];
            bool& nanImageVal = nanImageCudaPtr[index];
            densityVal = intensityVal == intensityVal ? 1.f : 0;
            nanImageVal = !static_cast<bool>(densityVal);
            if (nanImageVal) intensityVal = 0;
        });

        // Loop over the x, y and z dimensions
        for (int n = 0; n < 3; n++) {
            if (!axisToSmooth[n] || image->dim[n] <= 1) continue;

            double temp;
            if (sigma[t] > 0) temp = sigma[t] / image->pixdim[n + 1]; // mm to voxel
            else temp = fabs(sigma[t]); // voxel-based if negative value
            int radius = 0;
            // Define the kernel size
            if (kernelType == MEAN_KERNEL || kernelType == LINEAR_KERNEL) {
                // Mean or linear filtering
                radius = static_cast<int>(temp);
            } else if (kernelType == GAUSSIAN_KERNEL) {
                // Gaussian kernel
                radius = static_cast<int>(temp * 3.0);
            } else if (kernelType == CUBIC_SPLINE_KERNEL) {
                // Spline kernel
                radius = static_cast<int>(temp * 2.0);
            } else {
                NR_FATAL_ERROR("Unknown kernel type");
            }
            if (radius <= 0) continue;

            // Allocate the kernel
            vector<float> kernel(2 * radius + 1);
            double kernelSum = 0;
            // Fill the kernel
            if (kernelType == CUBIC_SPLINE_KERNEL) {
                // Compute the Cubic Spline kernel
                for (int i = -radius; i <= radius; i++) {
                    // temp contains the kernel node spacing
                    double relative = fabs(i / temp);
                    if (relative < 1.0)
                        kernel[i + radius] = static_cast<float>(2.0 / 3.0 - Square(relative) + 0.5 * Cube(relative));
                    else if (relative < 2.0)
                        kernel[i + radius] = static_cast<float>(-Cube(relative - 2.0) / 6.0);
                    else kernel[i + radius] = 0;
                    kernelSum += kernel[i + radius];
                }
            } else if (kernelType == GAUSSIAN_KERNEL) {
                // Compute the Gaussian kernel
                for (int i = -radius; i <= radius; i++) {
                    // 2.506... = sqrt(2*pi)
                    // temp contains the sigma in voxel
                    kernel[i + radius] = static_cast<float>(exp(-Square(i) / (2.0 * Square(temp))) / (temp * 2.506628274631));
                    kernelSum += kernel[i + radius];
                }
            } else if (kernelType == LINEAR_KERNEL) {
                // Compute the linear kernel
                for (int i = -radius; i <= radius; i++) {
                    kernel[i + radius] = 1.f - fabs(i / static_cast<float>(radius));
                    kernelSum += kernel[i + radius];
                }
            } else if (kernelType == MEAN_KERNEL && imageDims.z == 1) {
                // Compute the mean kernel
                for (int i = -radius; i <= radius; i++) {
                    kernel[i + radius] = 1.f;
                    kernelSum += kernel[i + radius];
                }
            }
            // No kernel is required for the mean filtering
            // No need for kernel normalisation as this is handled by the density function
            NR_DEBUG("Convolution type[" << kernelType << "] dim[" << n << "] tp[" << t << "] radius[" << radius << "] kernelSum[" << kernelSum << "]");

            int planeCount, lineOffset;
            switch (n) {
            case 0:
                planeCount = imageDims.y * imageDims.z;
                lineOffset = 1;
                break;
            case 1:
                planeCount = imageDims.x * imageDims.z;
                lineOffset = imageDims.x;
                break;
            case 2:
                planeCount = imageDims.x * imageDims.y;
                lineOffset = planeCount;
                break;
            }

            thrust::device_vector<float> kernelCuda(kernel.begin(), kernel.end());
            float *kernelCudaPtr = kernelCuda.data().get();
            const int imageDim = reinterpret_cast<const int*>(&imageDims)[n];

            // Loop over the different voxel
            thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), planeCount, [=]__device__(const int planeIndex) {
                int realIndex = 0;
                switch (n) {
                case 0:
                    realIndex = planeIndex * imageDims.x;
                    break;
                case 1:
                    realIndex = (planeIndex / imageDims.x) * imageDims.x * imageDims.y + planeIndex % imageDims.x;
                    break;
                case 2:
                    realIndex = planeIndex;
                    break;
                }
                // Fetch the current line into a stack buffer
                float *bufferIntensityPtr = &bufferIntensityCudaPtr[planeIndex * imageDim];
                float *bufferDensityPtr = &bufferDensityCudaPtr[planeIndex * imageDim];
                float4 *currentIntensityPtr = &imageCuda[realIndex];
                float *currentDensityPtr = &densityCudaPtr[realIndex];
                for (int lineIndex = 0; lineIndex < imageDim; ++lineIndex) {
                    bufferIntensityPtr[lineIndex] = reinterpret_cast<float*>(currentIntensityPtr)[t];
                    bufferDensityPtr[lineIndex] = *currentDensityPtr;
                    currentIntensityPtr += lineOffset;
                    currentDensityPtr += lineOffset;
                }
                if (kernelSum > 0) {
                    // Perform the kernel convolution along 1 line
                    for (int lineIndex = 0; lineIndex < imageDim; ++lineIndex) {
                        // Define the kernel boundaries
                        int shiftPre = lineIndex - radius;
                        int shiftPst = lineIndex + radius + 1;
                        float *kernelPtr;
                        if (shiftPre < 0) {
                            kernelPtr = &kernelCudaPtr[-shiftPre];
                            shiftPre = 0;
                        } else kernelPtr = kernelCudaPtr;
                        if (shiftPst > imageDim) shiftPst = imageDim;
                        // Set the current values to zero
                        // Increment the current value by performing the weighted sum
                        double intensitySum = 0, densitySum = 0;
                        for (int k = shiftPre; k < shiftPst; ++k) {
                            float& kernelValue = *kernelPtr++;
                            intensitySum += kernelValue * bufferIntensityPtr[k];
                            densitySum += kernelValue * bufferDensityPtr[k];
                        }
                        // Store the computed value in place
                        reinterpret_cast<float*>(&imageCuda[realIndex])[t] = static_cast<float>(intensitySum);
                        densityCudaPtr[realIndex] = static_cast<float>(densitySum);
                        realIndex += lineOffset;
                    } // line convolution
                } else { // kernelSum <= 0
                    for (int lineIndex = 1; lineIndex < imageDim; ++lineIndex) {
                        bufferIntensityPtr[lineIndex] += bufferIntensityPtr[lineIndex - 1];
                        bufferDensityPtr[lineIndex] += bufferDensityPtr[lineIndex - 1];
                    }
                    int shiftPre = -radius - 1;
                    int shiftPst = radius;
                    for (int lineIndex = 0; lineIndex < imageDim; ++lineIndex, ++shiftPre, ++shiftPst) {
                        float bufferIntensityCur, bufferDensityCur;
                        if (shiftPre > -1) {
                            if (shiftPst < imageDim) {
                                bufferIntensityCur = bufferIntensityPtr[shiftPre] - bufferIntensityPtr[shiftPst];
                                bufferDensityCur = bufferDensityPtr[shiftPre] - bufferDensityPtr[shiftPst];
                            } else {
                                bufferIntensityCur = bufferIntensityPtr[shiftPre] - bufferIntensityPtr[imageDim - 1];
                                bufferDensityCur = bufferDensityPtr[shiftPre] - bufferDensityPtr[imageDim - 1];
                            }
                        } else {
                            if (shiftPst < imageDim) {
                                bufferIntensityCur = -bufferIntensityPtr[shiftPst];
                                bufferDensityCur = -bufferDensityPtr[shiftPst];
                            } else {
                                bufferIntensityCur = 0;
                                bufferDensityCur = 0;
                            }
                        }
                        reinterpret_cast<float*>(&imageCuda[realIndex])[t] = bufferIntensityCur;
                        densityCudaPtr[realIndex] = bufferDensityCur;
                        realIndex += lineOffset;
                    } // line convolution of mean filter
                } // No kernel computation
            }); // pixel in starting plane
        } // axes

        // Normalise per time point
        thrust::for_each_n(thrust::device, thrust::make_counting_iterator<size_t>(0), voxelNumber, [=]__device__(const size_t index) {
            float& intensityVal = reinterpret_cast<float*>(&imageCuda[index])[t];
            const float& densityVal = densityCudaPtr[index];
            const bool& nanImageVal = nanImageCudaPtr[index];
            intensityVal = nanImageVal ? std::numeric_limits<float>::quiet_NaN() : intensityVal / densityVal;
        });
    } // check if the time point is active
}
/* *************************************************************** */
