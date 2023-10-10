/*
 *  reg_tools.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_blockMatching.h"
#include "_reg_globalTrans.h"
#include "_reg_localTrans.h"
#include "_reg_tools.h"
#include "_reg_mind.h"

#include "_reg_blockMatching.h"
#include "BlockMatchingKernel.h"
#include "Platform.h"
#include "AladinContent.h"

#include "reg_tools.h"

std::vector<float> splitFloatVector(char* input)
{
    std::vector<float> floatVector;
    char* charArray = strtok(input, ",");
    while (charArray != nullptr)
    {
        floatVector.push_back(atof(charArray));
        charArray = strtok(nullptr, ",");
    }

    return floatVector;
}

int isNumeric (const char *s)
{
    if(s==nullptr || *s=='\0' || isspace(*s))
        return EXIT_SUCCESS;
    char * p;
    strtod (s, &p);
    return *p == '\0';
}

typedef struct
{
        char *inputImageName;
        char *outputImageName;
        char *operationImageName;
        char *rmsImageName;
        float operationValue;
        float smoothValueX;
        float smoothValueY;
        float smoothValueZ;
        float thresholdImageValue;
        float removeNanInfValue;
        float pixdimX;
        float pixdimY;
        float pixdimZ;
        int interpOrder;
} PARAM;
typedef struct
{
        bool inputImageFlag;
        bool outputImageFlag;
        bool floatFlag;
        bool downsampleFlag;
        bool rmsImageFlag;
        bool smoothSplineFlag;
        bool smoothGaussianFlag;
        bool smoothLabFlag;
        bool smoothMeanFlag;
        bool binarisedImageFlag;
        bool thresholdImageFlag;
        bool nanMaskFlag;
        bool normFlag;
        int operationTypeFlag;
        bool iso;
        bool nosclFlag;
        bool removeNanInf;
        bool changeResFlag;
        bool rgbFlag;
        bool bsi2rgbFlag;
        bool testActiveBlocksFlag;
        bool mindFlag;
        bool mindSSCFlag;
        bool interpFlag;
} FLAG;


void PetitUsage(char *exec)
{
    NR_INFO("Usage:\t" << exec << " -in  <filename> [OPTIONS]");
    NR_INFO("\tSee the help for more details (-h)");
}

void Usage(char *exec)
{
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Usage:\t" << exec << " -in <filename> -out <filename> [OPTIONS]");
    NR_INFO("\t-in <filename>\tFilename of the input image image (mandatory)");
    NR_INFO("* * OPTIONS * *");
    NR_INFO("\t-out <filename>\t\tFilename out the output image [output.nii]");
    NR_INFO("\t-float\t\t\tThe input image is converted to float");
    NR_INFO("\t-down\t\t\tThe input image is downsampled 2 times");
    NR_INFO("\t-smoS <float> <float> <float>\n\t\t\t\tThe input image is smoothed using a cubic b-spline kernel");
    NR_INFO("\t-smoG <float> <float> <float>\n\t\t\t\tThe input image is smoothed using Gaussian kernel");
    NR_INFO("\t-smoL <float> <float> <float>\n\t\t\t\tThe input label image is smoothed using Gaussian kernel");
    NR_INFO("\t-add <filename/float>\tThis image (or value) is added to the input");
    NR_INFO("\t-sub <filename/float>\tThis image (or value) is subtracted to the input");
    NR_INFO("\t-mul <filename/float>\tThis image (or value) is multiplied to the input");
    NR_INFO("\t-div <filename/float>\tThis image (or value) is divided to the input");
    NR_INFO("\t-rms <filename>\t\tCompute the mean rms between both image");
    NR_INFO("\t-bin \t\t\tBinarise the input image (val!=0?val=1:val=0)");
    NR_INFO("\t-thr <float>\t\tThreshold the input image (val<thr?val=0:val=1)");
    NR_INFO("\t-nan <filename>\t\tThis image is used to mask the input image.\n\t\t\t\tVoxels outside of the mask are set to nan");
    NR_INFO("\t-iso\t\t\tThe resulting image is made isotropic");
    NR_INFO("\t-chgres <float> <float> <float>\n\t\t\t\tResample the input image to the specified resolution (in mm)");
    NR_INFO("\t-noscl\t\t\tThe scl_slope and scl_inter are set to 1 and 0 respectively");
    NR_INFO("\t-rmNanInf <float>\tRemove the nan and inf from the input image and replace them by the specified value");
    NR_INFO("\t-4d2rgb\t\t\tConvert a 4D (or 5D) to rgb nifti file");
    NR_INFO("\t-testActiveBlocks\tGenerate an image highlighting the active blocks for reg_aladin (block variance is shown)");
    NR_INFO("\t-mind\t\t\tCreate a MIND descriptor image");
    NR_INFO("\t-mindssc\t\tCreate a MIND-SSC descriptor image");
    NR_INFO("\t-interp\t\t\tInterpolation order to use to warp the floating image");
#ifdef _OPENMP
   int defaultOpenMPValue=omp_get_num_procs();
   if(getenv("OMP_NUM_THREADS")!=nullptr)
      defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
   NR_INFO("\t-omp <int>\t\tNumber of threads to use with OpenMP. [" << defaultOpenMPValue << "/" << omp_get_num_procs() << "]");
#endif
   NR_INFO("\t--version\t\tPrint current version and exit (" << NR_VERSION << ")");
   NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

int main(int argc, char **argv)
{
    PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
    FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
    flag->operationTypeFlag=-1;

    if (argc < 2)
    {
        PetitUsage(argv[0]);
        free(param);
        free(flag);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    // Set the default number of threads
    int defaultOpenMPValue=omp_get_num_procs();
    if(getenv("OMP_NUM_THREADS")!=nullptr)
        defaultOpenMPValue=atoi(getenv("OMP_NUM_THREADS"));
    omp_set_num_threads(defaultOpenMPValue);
#endif

    /* read the input parameter */
    for(int i=1; i<argc; i++)
    {
       if(strcmp(argv[i],"-h")==0 ||
             strcmp(argv[i],"-H")==0 ||
             strcmp(argv[i],"-help")==0 ||
             strcmp(argv[i],"--help")==0 ||
             strcmp(argv[i],"-HELP")==0 ||
             strcmp(argv[i],"--HELP")==0 ||
             strcmp(argv[i],"-Help")==0 ||
             strcmp(argv[i],"--Help")==0
         )
        {
            Usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if(strcmp(argv[i], "--xml")==0)
        {
            NR_COUT << xml_tools << std::endl;
            return EXIT_SUCCESS;
        }
        else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
        {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            NR_WARN("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
            ++i;
#endif
        }
        else if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
                strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
                strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0)
        {
            NR_COUT << NR_VERSION << std::endl;
            return EXIT_SUCCESS;
        }
        else if(strcmp(argv[i], "-in") == 0 || strcmp(argv[i], "--in") == 0)
        {
            param->inputImageName=argv[++i];
            flag->inputImageFlag=1;
        }
        else if(strcmp(argv[i], "-out") == 0 || strcmp(argv[i], "--out") == 0)
        {
            param->outputImageName=argv[++i];
            flag->outputImageFlag=1;
        }

        else if(strcmp(argv[i], "-add") == 0 || strcmp(argv[i], "--add") == 0)
        {
           char * val = argv[++i];
           if (isNumeric(val))
           {
              float floatVal = (float)atof(val);
              if(floatVal != -999999)
              {
                  param->operationValue=floatVal;
                  flag->operationTypeFlag=0;
              }
           }
           else
           {
             param->operationImageName=val;
             flag->operationTypeFlag=0;
           }
        }
        else if(strcmp(argv[i], "-sub") == 0 || strcmp(argv[i], "--sub") == 0)
        {
           char * val = argv[++i];
           if (isNumeric(val))
           {
              float floatVal = (float)atof(val);
              if(floatVal != -999999)
              {
                  param->operationValue=floatVal;
                  flag->operationTypeFlag=1;
              }
           }
           else
           {
             param->operationImageName=val;
             flag->operationTypeFlag=1;
           }
        }
        else if(strcmp(argv[i], "-mul") == 0 || strcmp(argv[i], "--mul") == 0)
        {
           char * val = argv[++i];
           if (isNumeric(val))
           {
              float floatVal = (float)atof(val);
              if(floatVal != -999999)
              {
                  param->operationValue=floatVal;
                  flag->operationTypeFlag=2;
              }
           }
           else
           {
             param->operationImageName=val;
             flag->operationTypeFlag=2;
           }
        }
        else if(strcmp(argv[i], "-iso") == 0 || strcmp(argv[i], "--iso") == 0)
        {
            flag->iso=true;
        }
        else if(strcmp(argv[i], "-div") == 0 || strcmp(argv[i], "--div") == 0)
        {
           char * val = argv[++i];
           if (isNumeric(val))
           {
              float floatVal = (float)atof(val);
              if(floatVal != -999999)
              {
                  param->operationValue=floatVal;
                  flag->operationTypeFlag=3;
              }
           }
           else
           {
             param->operationImageName=val;
             flag->operationTypeFlag=3;
           }
        }
        else if(strcmp(argv[i], "-rms") == 0 || strcmp(argv[i], "--rms") == 0)
        {
            param->rmsImageName=argv[++i];
            flag->rmsImageFlag=1;
        }
        else if(strcmp(argv[i], "-down") == 0 || strcmp(argv[i], "--down") == 0)
        {
            flag->downsampleFlag=1;
        }
        else if(strcmp(argv[i], "-float") == 0 || strcmp(argv[i], "--float") == 0)
        {
            flag->floatFlag=1;
        }
        else if(strcmp(argv[i], "-smoS") == 0 || strcmp(argv[i], "--smoS") == 0)
        {
          char* val = argv[++i];
          if (isNumeric(val))
          {
            param->smoothValueX=atof(val);
            param->smoothValueY=atof(argv[++i]);
            param->smoothValueZ=atof(argv[++i]);
            flag->smoothSplineFlag=1;
          }
          else
          {
            std::vector<float> valArray = splitFloatVector(val);
            if (valArray.size() == 3)
            {
                param->smoothValueX=valArray[0];
                param->smoothValueY=valArray[1];
                param->smoothValueZ=valArray[2];
                flag->smoothSplineFlag=1;
            }
          }
        }
        else if(strcmp(argv[i], "-smoG") == 0 || strcmp(argv[i], "--smoG") == 0)
        {
          char* val = argv[++i];
          if (isNumeric(val))
          {
            param->smoothValueX=atof(val);
            param->smoothValueY=atof(argv[++i]);
            param->smoothValueZ=atof(argv[++i]);
            flag->smoothGaussianFlag=1;
          }
          else
          {
            std::vector<float> valArray = splitFloatVector(val);
            if (valArray.size() == 3)
            {
                param->smoothValueX=valArray[0];
                param->smoothValueY=valArray[1];
                param->smoothValueZ=valArray[2];
                flag->smoothGaussianFlag=1;
            }
          }
        }
        else if(strcmp(argv[i], "-smoL") == 0 || strcmp(argv[i], "--smoL") == 0)
        {
          char* val = argv[++i];
          if (isNumeric(val))
          {
            param->smoothValueX=atof(val);
            param->smoothValueY=atof(argv[++i]);
            param->smoothValueZ=atof(argv[++i]);
            flag->smoothLabFlag=1;
          }
          else
          {
            std::vector<float> valArray = splitFloatVector(val);
            if (valArray.size() == 3)
            {
                param->smoothValueX=valArray[0];
                param->smoothValueY=valArray[1];
                param->smoothValueZ=valArray[2];
                flag->smoothLabFlag=1;
            }
          }
        }
        else if(strcmp(argv[i], "-smoM") == 0)
        {
            param->smoothValueX=atof(argv[++i]);
            param->smoothValueY=atof(argv[++i]);
            param->smoothValueZ=atof(argv[++i]);
            flag->smoothMeanFlag=1;
        }
        else if(strcmp(argv[i], "-bin") == 0 || strcmp(argv[i], "--bin") == 0)
        {
            flag->binarisedImageFlag=1;
        }
        else if(strcmp(argv[i], "-thr") == 0 || strcmp(argv[i], "--thr") == 0)
        {
            float val = atof(argv[++i]);
            if(val != -999999)
            {
                param->thresholdImageValue=val;
                flag->thresholdImageFlag=1;
            }
        }
        else if(strcmp(argv[i], "-nan") == 0 || strcmp(argv[i], "--nan") == 0)
        {
            param->operationImageName=argv[++i];
            flag->nanMaskFlag=1;
        }
        else if(strcmp(argv[i], "-norm") == 0)
        {
            flag->normFlag=1;
        }
        else if(strcmp(argv[i], "-noscl") == 0 || strcmp(argv[i], "--noscl") == 0)
        {
            flag->nosclFlag=1;
        }
        else if(strcmp(argv[i], "-rmNanInf") == 0 || strcmp(argv[i], "--rmNanInf") == 0)
        {
            float val = atof(argv[++i]);
            if(val != -999999)
            {
                flag->removeNanInf=1;
                param->removeNanInfValue=val;
            }
        }
        else if(strcmp(argv[i], "-chgres") == 0 || strcmp(argv[i], "--chgres") == 0)
        {
          char* val = argv[++i];
          if (isNumeric(val))
          {
              flag->changeResFlag=1;
              param->pixdimX=atof(val);
              param->pixdimY=atof(argv[++i]);
              param->pixdimZ=atof(argv[++i]);
          }
          else
          {
            std::vector<float> valArray = splitFloatVector(val);
            if (valArray.size() == 3)
            {
                param->pixdimX=valArray[0];
                param->pixdimY=valArray[1];
                param->pixdimZ=valArray[2];
                flag->changeResFlag=1;
            }
          }
        }
        else if(strcmp(argv[i], "-4d2rgb") == 0)
        {
            flag->rgbFlag=1;
        }
        else if(strcmp(argv[i], "-bsi2rgb") == 0)
        {
            flag->bsi2rgbFlag=1;
        }
        else if (strcmp(argv[i], "-testActiveBlocks") == 0 || strcmp(argv[i], "--testActiveBlocks") == 0)
        {
            flag->testActiveBlocksFlag=1;
        }
        else if(strcmp(argv[i], "-mind") == 0 || strcmp(argv[i], "--mind") == 0)
        {
            flag->mindFlag=1;
        }
        else if(strcmp(argv[i], "-mindssc") == 0 || strcmp(argv[i], "--mindssc") == 0)
        {
            flag->mindSSCFlag=1;
        }
        else if(strcmp(argv[i], "-interp") == 0 || strcmp(argv[i], "--interp") == 0)
        {
            flag->interpFlag=1;
            param->interpOrder=atoi(argv[++i]);
        }
        else
        {
            NR_ERROR("Unknown parameter: " << argv[i]);
            PetitUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    /* Read the image */
    nifti_image *image = reg_io_ReadImageFile(param->inputImageName);
    if(image == nullptr)
    {
        NR_ERROR("Error when reading the input image: " << param->inputImageName);
        return EXIT_FAILURE;
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->floatFlag)
    {
        reg_tools_changeDatatype<float>(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image, param->outputImageName);
        else reg_io_WriteImageFile(image, "output.nii");
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->downsampleFlag)
    {
        bool dim[8]= {true,true,true,true,true,true,true,true};
        reg_downsampleImage<float>(image, true, dim);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image, param->outputImageName);
        else reg_io_WriteImageFile(image, "output.nii");
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    // The image intensity are normalised between the 3 and 97%ile values
    if(flag->normFlag)
    {
        reg_tools_changeDatatype<float>(image);
        nifti_image *normImage = nifti_dup(*image);
        reg_heapSort(static_cast<float *>(normImage->data), normImage->nvox);
        float minValue = static_cast<float *>(normImage->data)[Floor(03*(int)normImage->nvox/100)];
        float maxValue = static_cast<float *>(normImage->data)[Floor(97*(int)normImage->nvox/100)];
        reg_tools_subtractValueFromImage(image,normImage,minValue);
        reg_tools_divideValueToImage(normImage,normImage,maxValue-minValue);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(normImage, param->outputImageName);
        else reg_io_WriteImageFile(normImage, "output.nii");
        nifti_image_free(normImage);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->smoothGaussianFlag || flag->smoothSplineFlag || flag->smoothMeanFlag)
    {
        nifti_image *smoothImg = nifti_dup(*image);
        float *kernelSize = new float[smoothImg->nt*smoothImg->nu];
        bool *timePoint = new bool[smoothImg->nt*smoothImg->nu];
        for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) timePoint[i]=true;
        bool boolX[3]= {1,0,0};
        for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueX;
        if(flag->smoothMeanFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Mean,nullptr,timePoint,boolX);
        else if(flag->smoothSplineFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Cubic,nullptr,timePoint,boolX);
        else reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Gaussian,nullptr,timePoint,boolX);
        bool boolY[3]= {0,1,0};
        for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueY;
        if(flag->smoothMeanFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Mean,nullptr,timePoint,boolY);
        else if(flag->smoothSplineFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Cubic,nullptr,timePoint,boolY);
        else reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Gaussian,nullptr,timePoint,boolY);
        bool boolZ[3]= {0,0,1};
        for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueZ;
        if(flag->smoothMeanFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Mean,nullptr,timePoint,boolZ);
        else if(flag->smoothSplineFlag)
            reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Cubic,nullptr,timePoint,boolZ);
        else reg_tools_kernelConvolution(smoothImg,kernelSize,ConvKernelType::Gaussian,nullptr,timePoint,boolZ);
        delete []kernelSize;
        delete []timePoint;
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(smoothImg, param->outputImageName);
        else reg_io_WriteImageFile(smoothImg, "output.nii");
        nifti_image_free(smoothImg);
    }


    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->smoothLabFlag)
    {
        nifti_image *smoothImg = nifti_dup(*image);

        bool *timePoint = new bool[smoothImg->nt*smoothImg->nu];
        for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) timePoint[i]=true;

        float varX=param->smoothValueX;
        float varY=param->smoothValueY;
        float varZ=param->smoothValueZ;

        reg_tools_labelKernelConvolution(smoothImg,varX,varY,varZ,nullptr,timePoint);

        delete []timePoint;
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(smoothImg, param->outputImageName);
        else reg_io_WriteImageFile(smoothImg, "output.nii");
        nifti_image_free(smoothImg);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->operationTypeFlag>-1)
    {
        nifti_image *image2=nullptr;
        if(param->operationImageName!=nullptr)
        {
            image2 = reg_io_ReadImageFile(param->operationImageName);
            if(image2 == nullptr)
            {
                NR_ERROR("Error when reading the image: " << param->operationImageName);
                return EXIT_FAILURE;
            }
        }
        // Images are converted to the higher datatype
        if(image2!=nullptr){
            switch(image->datatype>image2->datatype?image->datatype:image2->datatype)
            {
            case NIFTI_TYPE_UINT8:
                reg_tools_changeDatatype<unsigned char>(image,NIFTI_TYPE_UINT8);
                reg_tools_changeDatatype<unsigned char>(image2,NIFTI_TYPE_UINT8);
                break;
            case NIFTI_TYPE_INT8:
                reg_tools_changeDatatype<char>(image,NIFTI_TYPE_INT8);
                reg_tools_changeDatatype<char>(image2,NIFTI_TYPE_INT8);
                break;
            case NIFTI_TYPE_UINT16:
                reg_tools_changeDatatype<unsigned short>(image,NIFTI_TYPE_UINT16);
                reg_tools_changeDatatype<unsigned short>(image2,NIFTI_TYPE_UINT16);
                break;
            case NIFTI_TYPE_INT16:
                reg_tools_changeDatatype<short>(image,NIFTI_TYPE_INT16);
                reg_tools_changeDatatype<short>(image2,NIFTI_TYPE_INT16);
                break;
            case NIFTI_TYPE_UINT32:
                reg_tools_changeDatatype<unsigned>(image,NIFTI_TYPE_UINT32);
                reg_tools_changeDatatype<unsigned>(image2,NIFTI_TYPE_UINT32);
                break;
            case NIFTI_TYPE_INT32:
                reg_tools_changeDatatype<int>(image,NIFTI_TYPE_INT32);
                reg_tools_changeDatatype<int>(image2,NIFTI_TYPE_INT32);
                break;
            case NIFTI_TYPE_FLOAT32:
                reg_tools_changeDatatype<float>(image,NIFTI_TYPE_FLOAT32);
                reg_tools_changeDatatype<float>(image2,NIFTI_TYPE_FLOAT32);
                break;
            case NIFTI_TYPE_FLOAT64:
                reg_tools_changeDatatype<double>(image,NIFTI_TYPE_FLOAT64);
                reg_tools_changeDatatype<double>(image2,NIFTI_TYPE_FLOAT64);
                break;
            default:
                NR_ERROR("Unsupported data type!");
                return EXIT_FAILURE;
            }
        }

        nifti_image *outputImage = nifti_dup(*image, false);

        if(image2!=nullptr)
        {
            switch(flag->operationTypeFlag)
            {
            case 0:
                reg_tools_addImageToImage(image, image2, outputImage);
                break;
            case 1:
                reg_tools_subtractImageFromImage(image, image2, outputImage);
                break;
            case 2:
                reg_tools_multiplyImageToImage(image, image2, outputImage);
                break;
            case 3:
                reg_tools_divideImageToImage(image, image2, outputImage);
                break;
            }
        }
        else
        {
            switch(flag->operationTypeFlag)
            {
            case 0:
                reg_tools_addValueToImage(image, outputImage, param->operationValue);
                break;
            case 1:
                reg_tools_subtractValueFromImage(image, outputImage, param->operationValue);
                break;
            case 2:
                reg_tools_multiplyValueToImage(image, outputImage, param->operationValue);
                break;
            case 3:
                reg_tools_divideValueToImage(image, outputImage, param->operationValue);
                break;
            }
        }
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");

        nifti_image_free(outputImage);
        if(image2!=nullptr) nifti_image_free(image2);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->rmsImageFlag)
    {
        nifti_image *image2 = reg_io_ReadImageFile(param->rmsImageName);
        if(image2 == nullptr)
        {
            NR_ERROR("Error when reading the image: " << param->rmsImageName);
            return EXIT_FAILURE;
        }
        // Check image dimension
        if(image->dim[0]!=image2->dim[0] ||
                image->dim[1]!=image2->dim[1] ||
                image->dim[2]!=image2->dim[2] ||
                image->dim[3]!=image2->dim[3] ||
                image->dim[4]!=image2->dim[4] ||
                image->dim[5]!=image2->dim[5] ||
                image->dim[6]!=image2->dim[6] ||
                image->dim[7]!=image2->dim[7])
        {
            NR_ERROR("Both images do not have the same dimension");
            return EXIT_FAILURE;
        }

        double meanRMSerror = reg_tools_getMeanRMS(image, image2);
        NR_COUT << "Mean RMS error: " << meanRMSerror << std::endl;
        nifti_image_free(image2);
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->binarisedImageFlag)
    {
        reg_tools_binarise_image(image);
        reg_tools_changeDatatype<unsigned char>(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->thresholdImageFlag)
    {
        reg_tools_binarise_image(image, param->thresholdImageValue);
        reg_tools_changeDatatype<unsigned char>(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->nanMaskFlag)
    {
        nifti_image *maskImage = reg_io_ReadImageFile(param->operationImageName);
        if(maskImage == nullptr)
        {
            NR_ERROR("Error when reading the image: " << param->operationImageName);
            return EXIT_FAILURE;
        }

        nifti_image *outputImage = nifti_dup(*image, false);

        reg_tools_nanMask_image(image,maskImage,outputImage);

        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");

        nifti_image_free(outputImage);
        nifti_image_free(maskImage);
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->iso)
    {
        nifti_image *outputImage = reg_makeIsotropic(image,3);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->nosclFlag)
    {
        reg_tools_removeSCLInfo(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->removeNanInf)
    {
        size_t nanNumber=0, infNumber=0,finNumber=0;
        if(image->datatype==NIFTI_TYPE_FLOAT32)
        {
            float *imgDataPtr=static_cast<float *>(image->data);
            for(size_t i=0;i<image->nvox;++i){
                float value=imgDataPtr[i];
                if(value!=value){
                    nanNumber++;
                    imgDataPtr[i]=param->removeNanInfValue;
                }
                else if(value==std::numeric_limits<float>::infinity()){
                    infNumber++;
                    imgDataPtr[i]=param->removeNanInfValue;
                }
                else finNumber++;
            }
        }
        else if(image->datatype==NIFTI_TYPE_FLOAT64)
        {
            double *imgDataPtr=static_cast<double *>(image->data);
            for(size_t i=0;i<image->nvox;++i){
                double value=imgDataPtr[i];
                if(value!=value){
                    nanNumber++;
                    imgDataPtr[i]=param->removeNanInfValue;
                }
                else if(value==std::numeric_limits<double>::infinity()){
                    infNumber++;
                    imgDataPtr[i]=param->removeNanInfValue;
                }
                else finNumber++;
            }
        }
        else{
            NR_ERROR("Nan and Inf value can only be removed when the input image is of float or double datatype");
            return EXIT_FAILURE;
        }
        NR_COUT << "The input image contained " << nanNumber << " NaN, " << infNumber << " Inf and " << finNumber << " finite values" << std::endl;
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->changeResFlag)
    {
        // Define the size of the new image
        int newDim[8];
        for(size_t i=0; i<8; ++i) newDim[i]=image->dim[i];
        newDim[1]=Ceil((float)image->dim[1]*image->pixdim[1]/param->pixdimX);
        newDim[2]=Ceil((float)image->dim[2]*image->pixdim[2]/param->pixdimY);
        if(image->nz>1)
            newDim[3]=Ceil((float)image->dim[3]*image->pixdim[3]/param->pixdimZ);
        // Create the new image
        nifti_image *newImg=nifti_make_new_nim(newDim,image->datatype,true);
        newImg->pixdim[1]=newImg->dx=param->pixdimX;
        newImg->pixdim[2]=newImg->dy=param->pixdimY;
        if(image->nz>1)
            newImg->pixdim[3]=newImg->dz=param->pixdimZ;
        newImg->qform_code=image->qform_code;
        newImg->sform_code=image->sform_code;
        // Update the qform matrix
        newImg->qfac=image->qfac;
        newImg->quatern_b=image->quatern_b;
        newImg->quatern_c=image->quatern_c;
        newImg->quatern_d=image->quatern_d;
        newImg->qoffset_x=image->qoffset_x+newImg->dx/2.f-image->dx/2.f;
        newImg->qoffset_y=image->qoffset_y+newImg->dy/2.f-image->dy/2.f;
        if(image->nz>1)
            newImg->qoffset_z=image->qoffset_z+newImg->dz/2.f-image->dz/2.f;
        else newImg->qoffset_z=image->qoffset_z;
        newImg->qto_xyz=nifti_quatern_to_mat44(newImg->quatern_b,
                                               newImg->quatern_c,
                                               newImg->quatern_d,
                                               newImg->qoffset_x,
                                               newImg->qoffset_y,
                                               newImg->qoffset_z,
                                               newImg->pixdim[1],
                newImg->pixdim[2],
                newImg->pixdim[3],
                newImg->qfac);
        newImg->qto_ijk=nifti_mat44_inverse(newImg->qto_xyz);
        if(newImg->sform_code>0)
        {
            // Compute the new sform
            float scalingRatio[3];
            scalingRatio[0]= newImg->dx / image->dx;
            scalingRatio[1]= newImg->dy / image->dy;
            if(image->nz>1)
                scalingRatio[2]= newImg->dz / image->dz;
            else scalingRatio[2]=1.f;
            newImg->sto_xyz.m[0][0]=image->sto_xyz.m[0][0] * scalingRatio[0];
            newImg->sto_xyz.m[1][0]=image->sto_xyz.m[1][0] * scalingRatio[0];
            newImg->sto_xyz.m[2][0]=image->sto_xyz.m[2][0] * scalingRatio[0];
            newImg->sto_xyz.m[3][0]=image->sto_xyz.m[3][0];
            newImg->sto_xyz.m[0][1]=image->sto_xyz.m[0][1] * scalingRatio[1];
            newImg->sto_xyz.m[1][1]=image->sto_xyz.m[1][1] * scalingRatio[1];
            newImg->sto_xyz.m[2][1]=image->sto_xyz.m[2][1] * scalingRatio[1];
            newImg->sto_xyz.m[3][1]=image->sto_xyz.m[3][1];
            newImg->sto_xyz.m[0][2]=image->sto_xyz.m[0][2] * scalingRatio[2];
            newImg->sto_xyz.m[1][2]=image->sto_xyz.m[1][2] * scalingRatio[2];
            newImg->sto_xyz.m[2][2]=image->sto_xyz.m[2][2] * scalingRatio[2];
            newImg->sto_xyz.m[3][2]=image->sto_xyz.m[3][2];
            newImg->sto_xyz.m[0][3]=image->sto_xyz.m[0][3]+newImg->dx/2.f-image->dx/2.f;
            newImg->sto_xyz.m[1][3]=image->sto_xyz.m[1][3]+newImg->dy/2.f-image->dy/2.f;
            if(image->nz>1)
                newImg->sto_xyz.m[2][3]=image->sto_xyz.m[2][3]+newImg->dz/2.f-image->dz/2.f;
            else newImg->sto_xyz.m[2][3]=image->sto_xyz.m[2][3];
            newImg->sto_xyz.m[3][3]=image->sto_xyz.m[3][3];
            newImg->sto_ijk=nifti_mat44_inverse(newImg->sto_xyz);
        }
        reg_checkAndCorrectDimension(newImg);
        // Create a deformation field
        nifti_image *def=nifti_copy_nim_info(newImg);
        def->dim[0]=def->ndim=5;
        def->dim[4]=def->nt=1;
        def->pixdim[4]=def->dt=1.f;
        if(newImg->nz==1)
            def->dim[5]=def->nu=2;
        else def->dim[5]=def->nu=3;
        def->pixdim[5]=def->du=1.f;
        def->dim[6]=def->nv=1;
        def->pixdim[6]=def->dv=1.f;
        def->dim[7]=def->nw=1;
        def->pixdim[7]=def->dw=1.f;
        def->nvox = NiftiImage::calcVoxelNumber(def, def->ndim);
        def->nbyper = sizeof(float);
        def->datatype = NIFTI_TYPE_FLOAT32;
        def->data = calloc(def->nvox,def->nbyper);
        // Fill the deformation field with an identity transformation
        reg_getDeformationFromDisplacement(def);
        // Allocate and compute the Jacobian matrices
        const size_t jacobianVoxelNumber = NiftiImage::calcVoxelNumber(def, 3);
        mat33 *jacobian = (mat33 *)malloc(jacobianVoxelNumber * sizeof(mat33));
        for (size_t i = 0; i < jacobianVoxelNumber; ++i)
            reg_mat33_eye(&jacobian[i]);
        // resample the original image into the space of the new image
        if(flag->interpFlag == 0){
            param->interpOrder = 3;
        }
        //
        if((newImg->pixdim[1]>image->pixdim[1] ||
                newImg->pixdim[2]>image->pixdim[2] ||
                newImg->pixdim[3]>image->pixdim[3]) && param->interpOrder != 0){
            reg_resampleImage_PSF(image,
                                  newImg,
                                  def,
                                  nullptr,
                                  param->interpOrder,
                                  0.f,
                                  jacobian,
                                  0);
            NR_DEBUG("PSF resampling completed");
        }
        else{
            reg_resampleImage(image,
                              newImg,
                              def,
                              nullptr,
                              param->interpOrder,
                              0.f);
            NR_DEBUG("Resampling completed");
        }
        free(jacobian);
        nifti_image_free(def);
        // Save and free the new iamge
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(newImg,param->outputImageName);
        else reg_io_WriteImageFile(newImg,"output.nii");
        nifti_image_free(newImg);
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->rgbFlag)
    {
        // Convert the input image to float if needed
        if(image->datatype!=NIFTI_TYPE_FLOAT32)
            reg_tools_changeDatatype<float>(image);
        // Create a temporary scaled image
        nifti_image *scaledImage = nifti_dup(*image, false);
        // Rescale the input image
        float min_value = reg_tools_getMinValue(image, -1);
        float max_value = reg_tools_getMaxValue(image, -1);
        reg_tools_subtractValueFromImage(image, scaledImage, min_value);
        reg_tools_multiplyValueToImage(scaledImage, scaledImage, 255.f/(max_value-min_value));
        // Create the rgb image
        nifti_image *outputImage = nifti_copy_nim_info(image);
        outputImage->nt=outputImage->nu=outputImage->dim[4]=outputImage->dim[5]=1;
        outputImage->ndim=outputImage->dim[0]=outputImage->nz>1?3:2;
        outputImage->nvox = NiftiImage::calcVoxelNumber(outputImage, outputImage->ndim);
        outputImage->datatype = NIFTI_TYPE_RGB24;
        outputImage->nbyper = 3 * sizeof(unsigned char);
        outputImage->data = malloc(outputImage->nbyper*outputImage->nvox);
        // Convert the image
        float *inPtr = static_cast<float *>(scaledImage->data);
        unsigned char *outPtr = static_cast<unsigned char *>(outputImage->data);
        for(int t=0; t<image->nt*image->nu; ++t){
            for(int z=0; z<image->nz; ++z){
                for(int y=0; y<image->ny; ++y){
                    for(int x=0; x<image->nx; ++x){
                        size_t outIndex = ((z*image->ny+y)*image->nx+x)*image->nt*image->nu+t;
                        outPtr[outIndex] = Round(*inPtr);
                        ++inPtr;
                    }
                }
            }
        }
        // Free the scaled image
        nifti_image_free(scaledImage);
        scaledImage=nullptr;
        // Save the rgb image
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
        outputImage=nullptr;
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->bsi2rgbFlag)
    {
        // Convert the input image to float if needed
        if(image->datatype!=NIFTI_TYPE_FLOAT32)
            reg_tools_changeDatatype<float>(image);
        // Create the rgb image
        nifti_image *outputImage = nifti_copy_nim_info(image);
        outputImage->nt=outputImage->nu=outputImage->dim[4]=outputImage->dim[5]=1;
        outputImage->ndim=outputImage->dim[0]=outputImage->nz>1?3:2;
        outputImage->nvox = NiftiImage::calcVoxelNumber(outputImage, outputImage->ndim);
        outputImage->datatype = NIFTI_TYPE_RGB24;
        outputImage->nbyper = 3 * sizeof(unsigned char);
        outputImage->scl_slope = 1.f;
        outputImage->scl_inter = 0.f;
        outputImage->cal_min = 0.f;
        outputImage->cal_max = 255.f;
        outputImage->data = malloc(outputImage->nbyper*outputImage->nvox);
        // Convert the image
        float *inPtr = static_cast<float *>(image->data);
        unsigned char *outPtr = static_cast<unsigned char *>(outputImage->data);
        for(int z=0; z<image->nz; ++z){
            for(int y=0; y<image->ny; ++y){
                for(int x=0; x<image->nx; ++x){
                    float value = *inPtr * 255.f;
                    size_t outIndex = ((z*image->ny+y)*image->nx+x)*3;
                    if (value > 0)
                        outPtr[outIndex] = static_cast<unsigned char>(Round(value>255?255:value));
                    else outPtr[outIndex+1] = static_cast<unsigned char>(Round(-value<-255?-255:-value));
                    outPtr[outIndex+2] = 0;
                    ++inPtr;
                }
            }
        }
        // Save the rgb image
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
        outputImage=nullptr;
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->mindFlag)
    {
        if(image->ndim>3){
            NR_ERROR("MIND only support 2D or 3D image for now");
            return EXIT_FAILURE;
        }
        // Convert the input image to float if needed
        if(image->datatype!=NIFTI_TYPE_FLOAT32)
            reg_tools_changeDatatype<float>(image);
        // Create a output image
        nifti_image *outputImage = nifti_copy_nim_info(image);
        outputImage->dim[0]=outputImage->ndim=4;
        outputImage->dim[4]=outputImage->nt=image->nz>1?6:4;
        outputImage->nvox=(size_t)image->nvox*outputImage->nt;
        outputImage->data = malloc(outputImage->nvox * outputImage->nbyper);
        // Compute the MIND descriptor
        int *mask = (int *)calloc(image->nvox, sizeof(int));
        GetMindImageDescriptor(image, outputImage, mask, 1, 0);
        free(mask);
        // Save the MIND descriptor image
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
        outputImage=nullptr;
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->mindSSCFlag)
    {
        if(image->ndim>3){
            NR_ERROR("MIND-SSC only support 2D or 3D image for now");
            return EXIT_FAILURE;
        }
        // Convert the input image to float if needed
        if(image->datatype!=NIFTI_TYPE_FLOAT32)
            reg_tools_changeDatatype<float>(image);
        // Create a output image
        nifti_image *outputImage = nifti_copy_nim_info(image);
        outputImage->dim[0]=outputImage->ndim=4;
        outputImage->dim[4]=outputImage->nt=image->nz>1?12:4;
        outputImage->nvox=(size_t)image->nvox*outputImage->nt;
        outputImage->data = malloc(outputImage->nvox * outputImage->nbyper);
        // Compute the MIND-SSC descriptor
        int *mask = (int *)calloc(image->nvox, sizeof(int));
        GetMindSscImageDescriptor(image, outputImage, mask, 1, 0);
        free(mask);
        // Save the MIND descriptor image
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
        outputImage=nullptr;
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->testActiveBlocksFlag){
        // Convert the input image to float if needed
        if(image->datatype!=NIFTI_TYPE_FLOAT32)
            reg_tools_changeDatatype<float>(image);
        // Create a temporary mask
        const size_t voxelNumber = NiftiImage::calcVoxelNumber(image, 3);
        int *temp_mask = (int *)malloc(voxelNumber * sizeof(int));
        for (size_t i = 0; i < voxelNumber; ++i)
            temp_mask[i]=i;
        // Initialise the block matching
        _reg_blockMatchingParam bm_param;
        initialise_block_matching_method(image,
                                         &bm_param,
                                         100,
                                         100,
                                         1,
                                         temp_mask);


        // Generate an image to store the active blocks
        nifti_image *outputImage = nifti_copy_nim_info(image);
        outputImage->nt=outputImage->nu=outputImage->dim[4]=outputImage->dim[5]=1;
        outputImage->ndim=outputImage->dim[0]=outputImage->nz>1?3:2;
        outputImage->nvox = NiftiImage::calcVoxelNumber(outputImage, outputImage->ndim);
        outputImage->cal_min=0;
        outputImage->data = calloc(outputImage->nbyper, outputImage->nvox);
        float *inPtr = static_cast<float *>(image->data);
        float *outPtr = static_cast<float *>(outputImage->data);
        // Iterate through the blocks
        size_t blockIndex=0;
        for(size_t bz=0;bz<bm_param.blockNumber[2];++bz){
            size_t vz=4*bz;
            for(size_t by=0;by<bm_param.blockNumber[1];++by){
                size_t vy=4*by;
                for(size_t bx=0;bx<bm_param.blockNumber[0];++bx){
                    size_t vx=4*bx;
                    if(bm_param.totalBlock[blockIndex++]>-1){
                        float meanValue=0;
                        float activeVoxel=0;
                        for(size_t z=vz;z<vz+4;++z){
                            if(z<(size_t)outputImage->nz){
                                for(size_t y=vy;y<vy+4;++y){
                                    if(y<(size_t)outputImage->ny){
                                        size_t voxelIndex = (z*outputImage->ny+y)*outputImage->nx+vx;
                                        for(size_t x=vx;x<vx+4;++x){
                                            if(x<(size_t)outputImage->nx){
                                                meanValue += inPtr[voxelIndex];
                                                activeVoxel++;
                                            }
                                            voxelIndex++;
                                        } // x
                                    }
                                } // y
                            }
                        } // z
                        meanValue /= activeVoxel;
                        float variance=0;
                        for(size_t z=vz;z<vz+4;++z){
                            if(z<(size_t)outputImage->nz){
                                for(size_t y=vy;y<vy+4;++y){
                                    if(y<(size_t)outputImage->ny){
                                        size_t voxelIndex = (z*outputImage->ny+y)*outputImage->nx+vx;
                                        for(size_t x=vx;x<vx+4;++x){
                                            if(x<(size_t)outputImage->nx){
                                                variance += Square(meanValue - inPtr[voxelIndex]);
                                            }
                                            voxelIndex++;
                                        } // x
                                    }
                                } // y
                            }
                        } // z
                        variance /= activeVoxel;
                        for(size_t z=vz;z<vz+4;++z){
                            if(z<(size_t)outputImage->nz){
                                for(size_t y=vy;y<vy+4;++y){
                                    if(y<(size_t)outputImage->ny){
                                        size_t voxelIndex = (z*outputImage->ny+y)*outputImage->nx+vx;
                                        for(size_t x=vx;x<vx+4;++x){
                                            if(x<(size_t)outputImage->nx){
                                                outPtr[voxelIndex] = variance;
                                            }
                                            voxelIndex++;
                                        } // x
                                    }
                                } // y
                            }
                        } // z
                    } // active block
                } // bx
            } // by
        } // bz
        outputImage->cal_max=reg_tools_getMaxValue(outputImage, -1);

        free(temp_mask);

        // Save the output image
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(outputImage,param->outputImageName);
        else reg_io_WriteImageFile(outputImage,"output.nii");
        nifti_image_free(outputImage);
        outputImage=nullptr;
    }

    nifti_image_free(image);
    return EXIT_SUCCESS;
}
