/*
 *  reg_test_interp.cpp
 *
 *
 *  Created by Marc Modat on 10/05/2012.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "Eigen/unsupported/FFT"

#define SIZE 64
#define WIDTH 5

int main(int argc, char **argv)
{
    char msg[255];
    sprintf(msg,"Usage: %s dim type",argv[0]);
    if(argc!=3){
        reg_print_msg_error(msg);
        return EXIT_FAILURE;
    }
    const int dim=atoi(argv[1]);
    const int type=atoi(argv[2]);
    if(dim!=2 && dim!=3){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for dim are 2 and 3");
        return EXIT_FAILURE;
    }
    if(type!=0 && type!=1 && type!=2){
        reg_print_msg_error(msg);
        reg_print_msg_error("Expected value for type are 0, 1 and 3");
        return EXIT_FAILURE;
    }

    // Create two images
	int image_dim[8]={dim,SIZE,SIZE,dim==2?1:SIZE,1,1,1,1};
    nifti_image *image1=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    nifti_image *image2=nifti_make_new_nim(image_dim,NIFTI_TYPE_FLOAT32,true);
    reg_checkAndCorrectDimension(image1);
    reg_checkAndCorrectDimension(image2);

    // The floating image is filled with a cosine function
    float *img1Ptr = static_cast<float *>(image1->data);
    for(int z=0; z<image1->nz; ++z){
        for(int y=0; y<image1->ny; ++y){
			for(int x=0; x<image1->nx; ++x){
				*img1Ptr++=cos((float)x/(float)WIDTH) *
						cos((float)y/(float)WIDTH)*cos((float)z/(float)WIDTH);
            }
        }
    }
    memcpy(image2->data,image1->data,image2->nvox*image2->nbyper);

    // Both images are convolved with specified kernel
	float kernelWidth[1]={WIDTH};
    reg_tools_kernelConvolution(image1,kernelWidth,type);

//    // Convolution using the Fourrier space
    float *img2Ptr = static_cast<float *>(image2->data);
    Eigen::FFT<float> fft;
    for(size_t d=0;d<dim;++d){
        // Create the kernel to convolve
        std::vector<float> kernel;
        kernel.resize(image2->dim[d+1]);
        float kernelSum=0;
        for(size_t i=0;i<image2->dim[d+1];++i){
            float distToCenter = fabs((float)i - (float)image2->dim[d+1]/2.f);
            switch(type){
            case 0: // Gaussian kernel
				kernel[i]=exp(-reg_pow2(distToCenter)/(2.f*reg_pow2((float)WIDTH)))/((float)WIDTH*2.506628274631);
                break;
            case 1: // Spline kernel
				distToCenter /= (float)WIDTH;
                if(distToCenter<2.f){
                    if(distToCenter<1.f)
                        kernel[i]=(2.f/3.f - distToCenter*distToCenter +
                                   0.5f*distToCenter*distToCenter*distToCenter);
                    else kernel[i]=-(distToCenter-2.f)*(distToCenter-2.f)*(distToCenter-2.f)/6.f;
                }
                else kernel[i]=0;
                break;
            case 2: // Mean kernel
				kernel[i]=distToCenter<=WIDTH?1:0;
                break;
            }
            kernelSum += kernel[i];
        }
        // Normalise the kernel
        for(size_t i=0;i<image2->dim[d+1];++i){
            kernel[i] /= kernelSum;
        }
        // Convert the kernel to frequency space
        std::vector<std::complex<float> > freqKernel;
        fft.fwd(freqKernel,kernel);

        // Extract and convert every line
        size_t planeIndex, planeNumber, lineIndex, lineOffset, realIndex;
        switch(d){
        case 0:
            planeNumber=image2->dim[2]*image2->dim[3];
            lineOffset  = 1;
            break;
        case 1:
            planeNumber = image2->dim[1]*image2->dim[3];
            lineOffset  = image2->dim[1];
            break;
        case 2:
            planeNumber = image2->dim[1]*image2->dim[2];
            lineOffset  = planeNumber;
            break;
        }
        for(planeIndex=0; planeIndex<planeNumber; ++planeIndex){
            switch(d){
            case 0: realIndex = planeIndex * image2->dim[1];break;
            case 1: realIndex = (planeIndex/image2->dim[1]) *
                        image2->dim[1]*image2->dim[2] +
                        planeIndex%image2->dim[1]; break;
            case 2: realIndex = planeIndex;break;
            }
            // Fetch the current line
            std::vector<float> intensities;
            float *currentIntensityPtr= &img2Ptr[realIndex];
            for(lineIndex=0;lineIndex<image2->dim[d+1];++lineIndex){
                intensities.push_back(*currentIntensityPtr);
                currentIntensityPtr+=lineOffset;
            }
            std::vector<std::complex<float> > freqIntensity;
            fft.fwd(freqIntensity,intensities);
            // convolution in the frequency space
            std::vector<std::complex<float> >::iterator it1,it2;
            for(it1=freqIntensity.begin(),
                it2=freqKernel.begin();
                it1!=freqIntensity.end();
                ++it1,++it2){
                *it1 = *it2 * *it1;
            }
            fft.inv(intensities,freqIntensity);
            currentIntensityPtr= &img2Ptr[realIndex];
            for(lineIndex=0;lineIndex<image2->dim[d+1];++lineIndex){
                *currentIntensityPtr=intensities[lineIndex];
                currentIntensityPtr+=lineOffset;
            }
        }
    }

    // Compute the difference between both images
    img1Ptr = static_cast<float *>(image1->data);
    img2Ptr = static_cast<float *>(image2->data);
    float max_diff=0.f;
    for(size_t i=0; i<image1->nvox; ++i){
        float diff = fabs(*img1Ptr++ - *img2Ptr++);
        max_diff=max_diff>diff?max_diff:diff;
    }

    nifti_image_free(image1);
    nifti_image_free(image2);

    // Check if the test failed or passed
    if(max_diff>0.001f){
        fprintf(stderr, "Error: %g > %g\n",max_diff,0.001f);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
