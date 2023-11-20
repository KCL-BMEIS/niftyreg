/*
 *  reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 26/03/2009.
 *  Copyright (c) 2009-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

// OpenCL isn't supported!
#undef USE_OPENCL

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_f3d2.h"
#include "reg_f3d.h"
#include <float.h>
// #include <libgen.h> //DOES NOT WORK ON WINDOWS !

#ifdef _WIN32
#   include <time.h>
#endif

using PrecisionType = float;

void PetitUsage(char *exec) {
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Fast Free-Form Deformation algorithm for non-rigid registration");
    NR_INFO("Usage:\t" << exec << " -ref <referenceImageName> -flo <floatingImageName> [OPTIONS]");
    NR_INFO("\tSee the help for more details (-h)");
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

void Usage(char *exec) {
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Fast Free-Form Deformation (F3D) algorithm for non-rigid registration.");
    NR_INFO("Based on Modat et al., \"Fast Free-Form Deformation using");
    NR_INFO("graphics processing units\", CMPB, 2010");
    NR_INFO("For any comment, please contact Marc Modat (m.modat@ucl.ac.uk)");
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Usage:\t" << exec << " -ref <filename> -flo <filename> [OPTIONS]");
    NR_INFO("\t-ref <filename>\tFilename of the reference image (mandatory)");
    NR_INFO("\t-flo <filename>\tFilename of the floating image (mandatory)");
    NR_INFO("***************");
    NR_INFO("*** OPTIONS ***");
    NR_INFO("***************");
    NR_INFO("*** Initial transformation options (One option will be considered):");
    NR_INFO("\t-aff <filename>\t\tFilename which contains an affine transformation (Affine*Reference=Floating)");
    NR_INFO("\t-incpp <filename>\tFilename of the control point grid input");
    NR_INFO("\t\t\t\tThe coarse spacing is defined by this file.");
    NR_INFO("");
    NR_INFO("*** Output options:");
    NR_INFO("\t-cpp <filename>\t\tFilename of control point grid [outputCPP.nii]");
    NR_INFO("\t-res <filename> \tFilename of the resampled image [outputResult.nii]");
    NR_INFO("");
    NR_INFO("*** Input image options:");
    NR_INFO("\t-rmask <filename>\t\tFilename of a mask image in the reference space");
    NR_INFO("\t-smooR <float>\t\t\tSmooth the reference image using the specified sigma (mm) [0]");
    NR_INFO("\t-smooF <float>\t\t\tSmooth the floating image using the specified sigma (mm) [0]");
    NR_INFO("\t--rLwTh <float>\t\t\tLower threshold to apply to the reference image intensities [none]. Identical value for every time point.*");
    NR_INFO("\t--rUpTh <float>\t\t\tUpper threshold to apply to the reference image intensities [none]. Identical value for every time point.*");
    NR_INFO("\t--fLwTh <float>\t\t\tLower threshold to apply to the floating image intensities [none]. Identical value for every time point.*");
    NR_INFO("\t--fUpTh <float>\t\t\tUpper threshold to apply to the floating image intensities [none]. Identical value for every time point.*");
    NR_INFO("\t-rLwTh <tp> <float>\tLower threshold to apply to the reference image intensities [none]*");
    NR_INFO("\t-rUpTh <tp> <float>\tUpper threshold to apply to the reference image intensities [none]*");
    NR_INFO("\t-fLwTh <tp> <float>\tLower threshold to apply to the floating image intensities [none]*");
    NR_INFO("\t-fUpTh <tp> <float>\tUpper threshold to apply to the floating image intensities [none]*");
    NR_INFO("\t* The scl_slope and scl_inter from the nifti header are taken into account for the thresholds");
    NR_INFO("");
    NR_INFO("*** Spline options (All defined at full resolution):");
    NR_INFO("\t-sx <float>\t\tFinal grid spacing along the x axis in mm (in voxel if negative value) [5 voxels]");
    NR_INFO("\t-sy <float>\t\tFinal grid spacing along the y axis in mm (in voxel if negative value) [sx value]");
    NR_INFO("\t-sz <float>\t\tFinal grid spacing along the z axis in mm (in voxel if negative value) [sx value]");
    NR_INFO("");
    NR_INFO("*** Regularisation options:");
    NR_INFO("\t-be <float>\t\tWeight of the bending energy (second derivative of the transformation) penalty term [0.001]");
    NR_INFO("\t-le <float>\t\tWeight of first order penalty term (symmetric and anti-symmetric part of the Jacobian) [0.01]");
    NR_INFO("\t-jl <float>\t\tWeight of log of the Jacobian determinant penalty term [0.0]");
    NR_INFO("\t-noAppJL\t\tTo not approximate the JL value only at the control point position");
    NR_INFO("\t-land <float> <file>\tUse of a set of landmarks which distance should be minimised");
    NR_INFO("\t\t\t\tThe first argument corresponds to the weight given to this regularisation (between 0 and 1)");
    NR_INFO("\t\t\t\tThe second argument corresponds to a text file containing the landmark positions in millimetre as");
    NR_INFO("\t\t\t\t<refX> <refY> <refZ> <floX> <floY> <floZ>\\n for 3D images and");
    NR_INFO("\t\t\t\t<refX> <refY> <floX> <floY>\\n for 2D images");
    NR_INFO("");
    NR_INFO("*** Measure of similarity options:");
    NR_INFO("*** NMI with 64 bins is used except if specified otherwise");
    NR_INFO("\t--nmi\t\t\tNMI. Used NMI even when one or several other measures are specified");
    NR_INFO("\t--rbn <int>\t\tNMI. Number of bin to use for the reference image histogram. Identical value for every time point");
    NR_INFO("\t--fbn <int>\t\tNMI. Number of bin to use for the floating image histogram. Identical value for every time point");
    NR_INFO("\t-rbn <tp> <int>\t\tNMI. Number of bin to use for the reference image histogram for the specified time point");
    NR_INFO("\t-fbn <tp> <int>\t\tNMI. Number of bin to use for the floating image histogram for the specified time point");
    NR_INFO("\t--lncc <float>\t\tLNCC. Standard deviation of the Gaussian kernel. Identical value for every time point");
    NR_INFO("\t-lncc <tp> <float>\tLNCC. Standard deviation of the Gaussian kernel for the specified time point");
    NR_INFO("\t--ssd \t\t\tSSD. Used for all time points - images are normalized between 0 and 1 before computing the measure");
    NR_INFO("\t-ssd <tp> \t\tSSD. Used for the specified time point - images are normalized between 0 and 1 before computing the measure");
    NR_INFO("\t--ssdn \t\t\tSSD. Used for all time points - images are NOT normalized between 0 and 1 before computing the measure");
    NR_INFO("\t-ssdn <tp> \t\tSSD. Used for the specified time point - images are NOT normalized between 0 and 1 before computing the measure");
    NR_INFO("\t--mind <offset>\t\tMIND and the offset to use to compute the descriptor");
    NR_INFO("\t--mindssc <offset>\tMIND-SCC and the offset to use to compute the descriptor");
    NR_INFO("\t--kld\t\t\tKLD. Used for all time points");
    NR_INFO("\t-kld <tp>\t\tKLD. Used for the specified time point");
    NR_INFO("\t* For the Kullback-Leibler divergence, reference and floating are expected to be probabilities");
    NR_INFO("\t-rr\t\t\tIntensities are thresholded between the 2 and 98% ile");
    NR_INFO("*** Options for setting the weights for each time point for each similarity");
    NR_INFO("*** Note, the options above should be used first and will set a default weight of 1");
    NR_INFO("*** The options below should be used afterwards to set the desired weight if different to 1");
    NR_INFO("\t-nmiw <tp> <float>\tNMI Weight. Weight to use for the NMI similarity measure for the specified time point");
    NR_INFO("\t-lnccw <tp> <float>\tLNCC Weight. Weight to use for the LNCC similarity measure for the specified time point");
    NR_INFO("\t-ssdw <tp> <float>\tSSD Weight. Weight to use for the SSD similarity measure for the specified time point");
    NR_INFO("\t-kldw <tp> <float>\tKLD Weight. Weight to use for the KLD similarity measure for the specified time point");
    NR_INFO("\t-wSim <filename>\tWeight to apply to the measure of similarity at each voxel position");

    // NR_INFO("\t-amc\t\t\tTo use the additive NMI for multichannel data (bivariate NMI by default)");
    NR_INFO("");
    NR_INFO("*** Optimisation options:");
    NR_INFO("\t-maxit <int>\t\tMaximal number of iteration at the final level [150]");
    NR_INFO("\t-ln <int>\t\tNumber of level to perform [3]");
    NR_INFO("\t-lp <int>\t\tOnly perform the first levels [ln]");
    NR_INFO("\t-nopy\t\t\tDo not use a pyramidal approach");
    NR_INFO("\t-noConj\t\t\tTo not use the conjugate gradient optimisation but a simple gradient ascent");
    NR_INFO("\t-pert <int>\t\tTo add perturbation step(s) after each optimisation scheme");
    NR_INFO("");
    NR_INFO("*** F3D2 options:");
    NR_INFO("\t-vel \t\t\tUse a velocity field integration to generate the deformation");
    NR_INFO("\t-nogce \t\t\tDo not use the gradient accumulation through exponentiation");
    NR_INFO("\t-fmask <filename>\tFilename of a mask image in the floating space");
    NR_INFO("");

    if (Platform::IsCudaEnabled() || Platform::IsOpenClEnabled()) {
        NR_INFO("*** Platform options:");
        std::string platform = "\t-platf <uint>\t\tChoose platform: CPU=0 | ";
        if (Platform::IsCudaEnabled()) {
            platform += "Cuda=1";
            if (Platform::IsOpenClEnabled())
                platform += " | ";
        }
        if (Platform::IsOpenClEnabled())
            platform += "OpenCL=2";
        platform += " [0]";
        NR_INFO(platform);

        NR_INFO("\t-gpuid <uint>\t\tChoose a custom gpu.");
        NR_INFO("\t\t\t\tPlease run reg_gpuinfo first to get platform information and their corresponding ids");
    }

#ifdef _OPENMP
    NR_INFO("");
    NR_INFO("*** OpenMP-related options:");
    int defaultOpenMPValue = omp_get_num_procs();
    if (getenv("OMP_NUM_THREADS") != nullptr)
        defaultOpenMPValue = atoi(getenv("OMP_NUM_THREADS"));
    NR_INFO("\t-omp <int>\t\tNumber of threads to use with OpenMP. [" << defaultOpenMPValue << "/" << omp_get_num_procs() << "]");
#endif
    NR_INFO("");
    NR_INFO("*** Other options:");
    NR_INFO("\t-smoothGrad <float>\tTo smooth the metric derivative (in mm) [0]");
    NR_INFO("\t-pad <float>\t\tPadding value [nan]");
    NR_INFO("\t-voff\t\t\tTo turn verbose off");
    NR_INFO("\t--version\t\tPrint current version and exit");
    NR_INFO("\t\t\t\t(" << NR_VERSION << ")");
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

int main(int argc, char **argv) {
    if (argc == 1) {
        PetitUsage(argv[0]);
        return EXIT_FAILURE;
    }
    time_t start;
    time(&start);
    int verbose = true;

#ifdef _OPENMP
    // Set the default number of threads
    int defaultOpenMPValue = omp_get_num_procs();
    if (getenv("OMP_NUM_THREADS") != nullptr)
        defaultOpenMPValue = atoi(getenv("OMP_NUM_THREADS"));
    omp_set_num_threads(defaultOpenMPValue);
#endif

    std::string text;
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    // Check if any information is required
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 ||
            strcmp(argv[i], "-H") == 0 ||
            strcmp(argv[i], "-help") == 0 ||
            strcmp(argv[i], "--help") == 0 ||
            strcmp(argv[i], "-HELP") == 0 ||
            strcmp(argv[i], "--HELP") == 0 ||
            strcmp(argv[i], "-Help") == 0 ||
            strcmp(argv[i], "--Help") == 0
            ) {
            Usage((argv[0]));
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--xml") == 0) {
            NR_COUT << xml_f3d;
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "-voff") == 0) {
            NR_DEBUG("The verbose cannot be switch off in debug");
#ifdef NDEBUG
            verbose = false;
#endif
        }
        if (strcmp(argv[i], "-version") == 0 ||
            strcmp(argv[i], "-Version") == 0 ||
            strcmp(argv[i], "-V") == 0 ||
            strcmp(argv[i], "-v") == 0 ||
            strcmp(argv[i], "--v") == 0 ||
            strcmp(argv[i], "--version") == 0) {
            NR_COUT << NR_VERSION << std::endl;
            return EXIT_SUCCESS;
        }
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    // Output the command line
    PrintCmdLine(argc, argv, verbose);

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    // Read the reference and floating image
    NiftiImage referenceImage, floatingImage;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-ref") == 0) || (strcmp(argv[i], "-target") == 0) || (strcmp(argv[i], "--ref") == 0)) {
            referenceImage = reg_io_ReadImageFile(argv[++i]);
            if (!referenceImage) {
                NR_ERROR("Error when reading the reference image: " << argv[i - 1]);
                return EXIT_FAILURE;
            }
        }
        if ((strcmp(argv[i], "-flo") == 0) || (strcmp(argv[i], "-source") == 0) || (strcmp(argv[i], "--flo") == 0)) {
            floatingImage = reg_io_ReadImageFile(argv[++i]);
            if (!floatingImage) {
                NR_ERROR("Error when reading the floating image: " << argv[i - 1]);
                return EXIT_FAILURE;
            }
        }
    }
    // Check that both reference and floating image have been defined
    if (!referenceImage) {
        NR_ERROR("Error. No reference image has been defined");
        PetitUsage(argv[0]);
        return EXIT_FAILURE;
    }
    // Read the floating image
    if (!floatingImage) {
        NR_ERROR("Error. No floating image has been defined");
        PetitUsage(argv[0]);
        return EXIT_FAILURE;
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
    // Check the type of registration object to create
    unique_ptr<reg_f3d<PrecisionType>> reg;
    PlatformType platformType(PlatformType::Cpu);
    unsigned gpuIdx = 999;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-vel") == 0 || strcmp(argv[i], "--vel") == 0) {
            reg.reset(new reg_f3d2<PrecisionType>(referenceImage->nt, floatingImage->nt));
        } else if (strcmp(argv[i], "-platf") == 0 || strcmp(argv[i], "--platf") == 0) {
            PlatformType value{ atoi(argv[++i]) };
            if (value < PlatformType::Cpu || value > PlatformType::Cuda) {
                NR_ERROR("The platform argument is expected to be 0 or 1 | 0=CPU 1=CUDA");
                return EXIT_FAILURE;
            }
            if (value == PlatformType::Cuda && !Platform::IsCudaEnabled()) {
                NR_WARN("The current install of NiftyReg has not been compiled with CUDA");
                NR_WARN("The CPU platform is used");
                value = PlatformType::Cpu;
            }
            if (value == PlatformType::OpenCl && !Platform::IsOpenClEnabled()) {
                NR_WARN("The current install of NiftyReg has not been compiled with OpenCL");
                NR_WARN("The CPU platform is used");
                value = PlatformType::Cpu;
            }
            platformType = value;
        } else if (strcmp(argv[i], "-gpuid") == 0 || strcmp(argv[i], "--gpuid") == 0) {
            gpuIdx = unsigned(atoi(argv[++i]));
        }
    }
    if (!reg)
        reg.reset(new reg_f3d<PrecisionType>(referenceImage->nt, floatingImage->nt));
    reg->SetReferenceImage(referenceImage);
    reg->SetFloatingImage(floatingImage);
    reg->SetPlatformType(platformType);
    reg->SetGpuIdx(gpuIdx);

    // Create some pointers that could be used
    const char *outputWarpedImageName = "outputResult.nii";
    const char *outputCPPImageName = "outputCPP.nii";
    bool useMeanLNCC = false;
    int refBinNumber = 0;
    int floBinNumber = 0;

    /* read the input parameter */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ref") == 0 || strcmp(argv[i], "-target") == 0 ||
            strcmp(argv[i], "--ref") == 0 || strcmp(argv[i], "-flo") == 0 ||
            strcmp(argv[i], "-source") == 0 || strcmp(argv[i], "--flo") == 0 ||
            strcmp(argv[i], "-platf") == 0 || strcmp(argv[i], "--platf") == 0) {
            // argument has already been parsed
            ++i;
        } else if (strcmp(argv[i], "-voff") == 0) {
            verbose = false;
            reg->DoNotPrintOutInformation();
        } else if (strcmp(argv[i], "-aff") == 0 || (strcmp(argv[i], "--aff") == 0)) {
            // Check first if the specified affine file exist
            char *affineTransformationName = argv[++i];
            if (FILE *aff = fopen(affineTransformationName, "r")) {
                fclose(aff);
            } else {
                NR_ERROR("The specified input affine file can not be read: " << affineTransformationName);
                return EXIT_FAILURE;
            }
            // Read the affine matrix
            mat44 affineMatrix;
            reg_tool_ReadAffineFile(&affineMatrix, affineTransformationName);
            // Send the transformation to the registration object
            reg->SetAffineTransformation(affineMatrix);
        } else if (strcmp(argv[i], "-incpp") == 0 || (strcmp(argv[i], "--incpp") == 0)) {
            NiftiImage inputCCPImage = reg_io_ReadImageFile(argv[++i]);
            if (!inputCCPImage) {
                NR_ERROR("Error when reading the input control point grid image: " << argv[i - 1]);
                return EXIT_FAILURE;
            }
            reg->SetControlPointGridImage(std::move(inputCCPImage));
        } else if ((strcmp(argv[i], "-rmask") == 0) || (strcmp(argv[i], "-tmask") == 0) || (strcmp(argv[i], "--rmask") == 0)) {
            NiftiImage referenceMaskImage = reg_io_ReadImageFile(argv[++i]);
            if (!referenceMaskImage) {
                NR_ERROR("Error when reading the reference mask image: " << argv[i - 1]);
                return EXIT_FAILURE;
            }
            reg->SetReferenceMask(std::move(referenceMaskImage));
        } else if ((strcmp(argv[i], "-res") == 0) || (strcmp(argv[i], "-result") == 0) || (strcmp(argv[i], "--res") == 0)) {
            outputWarpedImageName = argv[++i];
        } else if (strcmp(argv[i], "-cpp") == 0 || (strcmp(argv[i], "--cpp") == 0)) {
            outputCPPImageName = argv[++i];
        } else if (strcmp(argv[i], "-maxit") == 0 || strcmp(argv[i], "--maxit") == 0) {
            reg->SetMaximalIterationNumber(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-sx") == 0 || strcmp(argv[i], "--sx") == 0) {
            reg->SetSpacing(0, (PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-sy") == 0 || strcmp(argv[i], "--sy") == 0) {
            reg->SetSpacing(1, (PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-sz") == 0 || strcmp(argv[i], "--sz") == 0) {
            reg->SetSpacing(2, (PrecisionType)atof(argv[++i]));
        } else if ((strcmp(argv[i], "--nmi") == 0)) {
            int bin = 64;
            if (refBinNumber != 0)
                bin = refBinNumber;
            for (int t = 0; t < referenceImage->nt; ++t)
                reg->UseNMISetReferenceBinNumber(t, bin);
            bin = 64;
            if (floBinNumber != 0)
                bin = floBinNumber;
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->UseNMISetFloatingBinNumber(t, bin);
        } else if ((strcmp(argv[i], "-rbn") == 0) || (strcmp(argv[i], "-tbn") == 0)) {
            int tp = atoi(argv[++i]);
            int bin = atoi(argv[++i]);
            refBinNumber = bin;
            reg->UseNMISetReferenceBinNumber(tp, bin);
        } else if ((strcmp(argv[i], "--rbn") == 0)) {
            int bin = atoi(argv[++i]);
            refBinNumber = bin;
            for (int t = 0; t < referenceImage->nt; ++t)
                reg->UseNMISetReferenceBinNumber(t, bin);
        } else if ((strcmp(argv[i], "-fbn") == 0) || (strcmp(argv[i], "-sbn") == 0)) {
            int tp = atoi(argv[++i]);
            int bin = atoi(argv[++i]);
            floBinNumber = bin;
            reg->UseNMISetFloatingBinNumber(tp, bin);
        } else if ((strcmp(argv[i], "--fbn") == 0)) {
            int bin = atoi(argv[++i]);
            floBinNumber = bin;
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->UseNMISetFloatingBinNumber(t, bin);
        } else if (strcmp(argv[i], "-ln") == 0 || strcmp(argv[i], "--ln") == 0) {
            reg->SetLevelNumber(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-lp") == 0 || strcmp(argv[i], "--lp") == 0) {
            reg->SetLevelToPerform(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-be") == 0 || strcmp(argv[i], "--be") == 0) {
            reg->SetBendingEnergyWeight((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-le") == 0 || strcmp(argv[i], "--le") == 0) {
            reg->SetLinearEnergyWeight((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-jl") == 0 || strcmp(argv[i], "--jl") == 0) {
            reg->SetJacobianLogWeight((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-noAppJL") == 0 || strcmp(argv[i], "--noAppJL") == 0) {
            reg->DoNotApproximateJacobianLog();
        } else if (strcmp(argv[i], "-land") == 0 || strcmp(argv[i], "--land") == 0) {
            float weight = (float)atof(argv[++i]);
            char *filename = argv[++i];
            std::pair<size_t, size_t> inputMatrixSize = reg_tool_sizeInputMatrixFile(filename);
            size_t landmarkNumber = inputMatrixSize.first;
            size_t n = inputMatrixSize.second;
            if (n == 4 && referenceImage->nz > 1) {
                NR_ERROR("4 values per line are expected for 2D images");
                return EXIT_FAILURE;
            } else if (n == 6 && referenceImage->nz < 2) {
                NR_ERROR("6 values per line are expected for 3D images");
                return EXIT_FAILURE;
            } else if (n != 4 && n != 6) {
                NR_ERROR("4 or 6 values are expected per line");
                return EXIT_FAILURE;
            }
            float **allLandmarks = reg_tool_ReadMatrixFile<float>(filename, landmarkNumber, n);
            unique_ptr<float[]> referenceLandmark(new float[landmarkNumber * n / 2]);
            unique_ptr<float[]> floatingLandmark(new float[landmarkNumber * n / 2]);
            for (size_t l = 0, index = 0; l < landmarkNumber; ++l) {
                referenceLandmark[index] = allLandmarks[l][0];
                referenceLandmark[index + 1] = allLandmarks[l][1];
                if (n == 4) {
                    floatingLandmark[index] = allLandmarks[l][2];
                    floatingLandmark[index + 1] = allLandmarks[l][3];
                    index += 2;
                } else {
                    referenceLandmark[index + 2] = allLandmarks[l][2];
                    floatingLandmark[index] = allLandmarks[l][3];
                    floatingLandmark[index + 1] = allLandmarks[l][4];
                    floatingLandmark[index + 2] = allLandmarks[l][5];
                    index += 3;
                }
            }
            reg->SetLandmarkRegularisationParam(landmarkNumber,
                                                referenceLandmark.get(),
                                                floatingLandmark.get(),
                                                weight);
            for (size_t l = 0; l < landmarkNumber; ++l)
                free(allLandmarks[l]);
            free(allLandmarks);
        } else if ((strcmp(argv[i], "-smooR") == 0) || (strcmp(argv[i], "-smooT") == 0) || strcmp(argv[i], "--smooR") == 0) {
            reg->SetReferenceSmoothingSigma((PrecisionType)atof(argv[++i]));
        } else if ((strcmp(argv[i], "-smooF") == 0) || (strcmp(argv[i], "-smooS") == 0) || strcmp(argv[i], "--smooF") == 0) {
            reg->SetFloatingSmoothingSigma((PrecisionType)atof(argv[++i]));
        } else if ((strcmp(argv[i], "-rLwTh") == 0) || (strcmp(argv[i], "-tLwTh") == 0)) {
            int tp = atoi(argv[++i]);
            PrecisionType val = (PrecisionType)atof(argv[++i]);
            reg->SetReferenceThresholdLow(tp, val);
        } else if ((strcmp(argv[i], "-rUpTh") == 0) || strcmp(argv[i], "-tUpTh") == 0) {
            int tp = atoi(argv[++i]);
            PrecisionType val = (PrecisionType)atof(argv[++i]);
            reg->SetReferenceThresholdUp(tp, val);
        } else if ((strcmp(argv[i], "-fLwTh") == 0) || (strcmp(argv[i], "-sLwTh") == 0)) {
            int tp = atoi(argv[++i]);
            PrecisionType val = (PrecisionType)atof(argv[++i]);
            reg->SetFloatingThresholdLow(tp, val);
        } else if ((strcmp(argv[i], "-fUpTh") == 0) || (strcmp(argv[i], "-sUpTh") == 0)) {
            int tp = atoi(argv[++i]);
            PrecisionType val = (PrecisionType)atof(argv[++i]);
            reg->SetFloatingThresholdUp(tp, val);
        } else if ((strcmp(argv[i], "--rLwTh") == 0)) {
            PrecisionType threshold = (PrecisionType)atof(argv[++i]);
            for (int t = 0; t < referenceImage->nt; ++t)
                reg->SetReferenceThresholdLow(t, threshold);
        } else if ((strcmp(argv[i], "--rUpTh") == 0)) {
            PrecisionType threshold = (PrecisionType)atof(argv[++i]);
            for (int t = 0; t < referenceImage->nt; ++t)
                reg->SetReferenceThresholdUp(t, threshold);
        } else if ((strcmp(argv[i], "--fLwTh") == 0)) {
            PrecisionType threshold = (PrecisionType)atof(argv[++i]);
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->SetFloatingThresholdLow(t, threshold);
        } else if ((strcmp(argv[i], "--fUpTh") == 0)) {
            PrecisionType threshold = (PrecisionType)atof(argv[++i]);
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->SetFloatingThresholdUp(t, threshold);
        } else if (strcmp(argv[i], "-smoothGrad") == 0) {
            reg->SetGradientSmoothingSigma((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "--smoothGrad") == 0) {
            reg->SetGradientSmoothingSigma((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-ssd") == 0) {
            int timePoint = atoi(argv[++i]);
            bool normalise = 1;
            reg->UseSSD(timePoint, normalise);
        } else if (strcmp(argv[i], "--ssd") == 0) {
            bool normalise = 1;
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->UseSSD(t, normalise);
        } else if (strcmp(argv[i], "-ssdn") == 0) {
            int timePoint = atoi(argv[++i]);
            bool normalise = 0;
            reg->UseSSD(timePoint, normalise);
        } else if (strcmp(argv[i], "--ssdn") == 0) {
            bool normalise = 0;
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->UseSSD(t, normalise);
        } else if (strcmp(argv[i], "--mind") == 0) {
            int offset = atoi(argv[++i]);
            if (offset != -999999) { // Value specified by the CLI - to be ignored
                if (referenceImage->nt > 1 || floatingImage->nt > 1) {
                    NR_ERROR("reg_mind does not support multiple time point image");
                    return EXIT_FAILURE;
                }
                reg->UseMIND(0, offset);
            }
        } else if (strcmp(argv[i], "--mindssc") == 0) {
            int offset = atoi(argv[++i]);
            if (offset != -999999) { // Value specified by the CLI - to be ignored
                if (referenceImage->nt > 1 || floatingImage->nt > 1) {
                    NR_ERROR("reg_mindssc does not support multiple time point image");
                    return EXIT_FAILURE;
                }
                reg->UseMINDSSC(0, offset);
            }
        } else if (strcmp(argv[i], "-kld") == 0) {
            reg->UseKLDivergence(atoi(argv[++i]));
        } else if (strcmp(argv[i], "--kld") == 0) {
            for (int t = 0; t < floatingImage->nt; ++t)
                reg->UseKLDivergence(t);
        } else if (strcmp(argv[i], "-rr") == 0 || strcmp(argv[i], "--rr") == 0) {
            reg->UseRobustRange();
        } else if (strcmp(argv[i], "-lncc") == 0) {
            int tp = atoi(argv[++i]);
            float stdev = (float)atof(argv[++i]);
            reg->UseLNCC(tp, stdev);
        } else if (strcmp(argv[i], "--lncc") == 0) {
            float stdev = (float)atof(argv[++i]);
            if (stdev != -999999) { // Value specified by the CLI - to be ignored
                for (int t = 0; t < referenceImage->nt; ++t)
                    reg->UseLNCC(t, stdev);
            }
        } else if (strcmp(argv[i], "-lnccMean") == 0) {
            useMeanLNCC = true;
        } else if (strcmp(argv[i], "-dti") == 0 || strcmp(argv[i], "--dti") == 0) {
            unique_ptr<bool[]> timePoint(new bool[referenceImage->nt]);
            for (int t = 0; t < referenceImage->nt; ++t)
                timePoint[t] = false;
            timePoint[atoi(argv[++i])] = true;
            timePoint[atoi(argv[++i])] = true;
            timePoint[atoi(argv[++i])] = true;
            if (referenceImage->nz > 1) {
                timePoint[atoi(argv[++i])] = true;
                timePoint[atoi(argv[++i])] = true;
                timePoint[atoi(argv[++i])] = true;
            }
            reg->UseDTI(timePoint.get());
        } else if (strcmp(argv[i], "-nmiw") == 0) {
            int tp = atoi(argv[++i]);
            double w = atof(argv[++i]);
            reg->SetNMIWeight(tp, w);
        } else if (strcmp(argv[i], "-lnccw") == 0) {
            int tp = atoi(argv[++i]);
            double w = atof(argv[++i]);
            reg->SetLNCCWeight(tp, w);
        } else if (strcmp(argv[i], "-ssdw") == 0) {
            int tp = atoi(argv[++i]);
            double w = atof(argv[++i]);
            reg->SetSSDWeight(tp, w);
        } else if (strcmp(argv[i], "-kldw") == 0) {
            int tp = atoi(argv[++i]);
            double w = atof(argv[++i]);
            reg->SetKLDWeight(tp, w);
        } else if (strcmp(argv[i], "-wSim") == 0 || strcmp(argv[i], "--wSim") == 0) {
            NiftiImage refLocalWeightSim = reg_io_ReadImageFile(argv[++i]);
            reg->SetLocalWeightSim(std::move(refLocalWeightSim));
        } else if (strcmp(argv[i], "-pad") == 0 || strcmp(argv[i], "--pad") == 0) {
            reg->SetWarpedPaddingValue((float)atof(argv[++i]));
        } else if (strcmp(argv[i], "-nopy") == 0 || strcmp(argv[i], "--nopy") == 0) {
            reg->DoNotUsePyramidalApproach();
        } else if (strcmp(argv[i], "-noConj") == 0 || strcmp(argv[i], "--noConj") == 0) {
            reg->DoNotUseConjugateGradient();
        } else if (strcmp(argv[i], "-approxGrad") == 0 || strcmp(argv[i], "--approxGrad") == 0) {
            reg->UseApproximatedGradient();
        } else if (strcmp(argv[i], "-interp") == 0 || strcmp(argv[i], "--interp") == 0) {
            int interp = atoi(argv[++i]);
            switch (interp) {
            case 0:
                reg->UseNearestNeighborInterpolation();
                break;
            case 1:
                reg->UseLinearInterpolation();
                break;
            default:
                reg->UseCubicSplineInterpolation();
                break;
            }
        } else if ((strcmp(argv[i], "-fmask") == 0) || (strcmp(argv[i], "-smask") == 0) ||
                 (strcmp(argv[i], "--fmask") == 0) || (strcmp(argv[i], "--smask") == 0)) {
            NiftiImage floatingMaskImage = reg_io_ReadImageFile(argv[++i]);
            if (!floatingMaskImage) {
                NR_ERROR("Error when reading the floating mask image: " << argv[i - 1]);
                return EXIT_FAILURE;
            }
            reg->SetFloatingMask(std::move(floatingMaskImage));
        } else if (strcmp(argv[i], "-ic") == 0 || strcmp(argv[i], "--ic") == 0) {
            reg->SetInverseConsistencyWeight((PrecisionType)atof(argv[++i]));
        } else if (strcmp(argv[i], "-nox") == 0) {
            reg->NoOptimisationAlongX();
        } else if (strcmp(argv[i], "-noy") == 0) {
            reg->NoOptimisationAlongY();
        } else if (strcmp(argv[i], "-noz") == 0) {
            reg->NoOptimisationAlongZ();
        } else if (strcmp(argv[i], "-pert") == 0 || strcmp(argv[i], "--pert") == 0) {
            reg->SetPerturbationNumber((size_t)atoi(argv[++i]));
        } else if (strcmp(argv[i], "-nogr") == 0) {
            reg->NoGridRefinement();
        } else if (strcmp(argv[i], "-nogce") == 0 || strcmp(argv[i], "--nogce") == 0) {
            reg->DoNotUseGradientCumulativeExp();
        } else if (strcmp(argv[i], "-bch") == 0 || strcmp(argv[i], "--bch") == 0) {
            reg->UseBCHUpdate(atoi(argv[++i]));
        }
        else if (strcmp(argv[i], "-omp") == 0 || strcmp(argv[i], "--omp") == 0) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            NR_WARN("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
            ++i;
#endif
        }
        /* All the following arguments should have already been parsed */
        else if (strcmp(argv[i], "-help") != 0 && strcmp(argv[i], "-Help") != 0 &&
                 strcmp(argv[i], "-HELP") != 0 && strcmp(argv[i], "-h") != 0 &&
                 strcmp(argv[i], "--h") != 0 && strcmp(argv[i], "--help") != 0 &&
                 strcmp(argv[i], "--xml") != 0 && strcmp(argv[i], "-version") != 0 &&
                 strcmp(argv[i], "-Version") != 0 && strcmp(argv[i], "-V") != 0 &&
                 strcmp(argv[i], "-v") != 0 && strcmp(argv[i], "--v") != 0 &&
                 strcmp(argv[i], "-platf") != 0 && strcmp(argv[i], "--platf") != 0 &&
                 strcmp(argv[i], "-vel") != 0) {
            NR_ERROR("\tUnknown parameter: " << argv[i]);
            PetitUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (useMeanLNCC)
        reg->SetLNCCKernelType(ConvKernelType::Gaussian);

    NR_DEBUG("*******************************************");
    NR_DEBUG("*******************************************");
    NR_DEBUG("NiftyReg has been compiled in DEBUG mode");
    NR_DEBUG("Please re-run cmake to set the variable");
    NR_DEBUG("CMAKE_BUILD_TYPE to \"Release\" if required");
    NR_DEBUG("*******************************************");
    NR_DEBUG("*******************************************");

#ifdef _OPENMP
    NR_VERBOSE_APP("OpenMP is used with " << omp_get_max_threads() << " threads");
#endif

    // Run the registration
    reg->Run();

    // Save the control point image
    NiftiImage outputControlPointGridImage = reg->GetControlPointPositionImage();
    memset(outputControlPointGridImage->descrip, 0, 80);
    strcpy(outputControlPointGridImage->descrip, "Control point position from NiftyReg (reg_f3d)");
    if (strcmp("NiftyReg F3D2", reg->GetExecutableName()) == 0)
        strcpy(outputControlPointGridImage->descrip, "Velocity field grid from NiftyReg (reg_f3d2)");
    reg_io_WriteImageFile(outputControlPointGridImage, outputCPPImageName);

    // Save the backward control point image
    if (reg->GetSymmetricStatus()) {
        // _backward is added to the forward control point grid image name
        std::string fname(outputCPPImageName);
        if (fname.find(".nii.gz") != std::string::npos)
            fname.replace(fname.find(".nii.gz"), 7, "_backward.nii.gz");
        else if (fname.find(".nii") != std::string::npos)
            fname.replace(fname.find(".nii"), 4, "_backward.nii");
        else if (fname.find(".hdr") != std::string::npos)
            fname.replace(fname.find(".hdr"), 4, "_backward.hdr");
        else if (fname.find(".img.gz") != std::string::npos)
            fname.replace(fname.find(".img.gz"), 7, "_backward.img.gz");
        else if (fname.find(".img") != std::string::npos)
            fname.replace(fname.find(".img"), 4, "_backward.img");
        else if (fname.find(".png") != std::string::npos)
            fname.replace(fname.find(".png"), 4, "_backward.png");
        else if (fname.find(".nrrd") != std::string::npos)
            fname.replace(fname.find(".nrrd"), 5, "_backward.nrrd");
        else fname.append("_backward.nii");
        NiftiImage outputBackwardControlPointGridImage = reg->GetBackwardControlPointPositionImage();
        memset(outputBackwardControlPointGridImage->descrip, 0, 80);
        strcpy(outputBackwardControlPointGridImage->descrip, "Backward Control point position from NiftyReg (reg_f3d)");
        if (strcmp("NiftyReg F3D2", reg->GetExecutableName()) == 0)
            strcpy(outputBackwardControlPointGridImage->descrip, "Backward velocity field grid from NiftyReg (reg_f3d2)");
        reg_io_WriteImageFile(outputBackwardControlPointGridImage, fname.c_str());
    }

    // Save the warped image(s)
    auto outputWarpedImages = reg->GetWarpedImage();
    memset(outputWarpedImages[0]->descrip, 0, 80);
    strcpy(outputWarpedImages[0]->descrip, "Warped image using NiftyReg (reg_f3d)");
    if (strcmp("NiftyReg F3D2", reg->GetExecutableName()) == 0) {
        strcpy(outputWarpedImages[0]->descrip, "Warped image using NiftyReg (reg_f3d2)");
        strcpy(outputWarpedImages[1]->descrip, "Warped image using NiftyReg (reg_f3d2)");
    }
    if (reg->GetSymmetricStatus()) {
        if (outputWarpedImages[1]) {
            std::string fname(outputWarpedImageName);
            if (fname.find(".nii.gz") != std::string::npos)
                fname.replace(fname.find(".nii.gz"), 7, "_backward.nii.gz");
            else if (fname.find(".nii") != std::string::npos)
                fname.replace(fname.find(".nii"), 4, "_backward.nii");
            else if (fname.find(".hdr") != std::string::npos)
                fname.replace(fname.find(".hdr"), 4, "_backward.hdr");
            else if (fname.find(".img.gz") != std::string::npos)
                fname.replace(fname.find(".img.gz"), 7, "_backward.img.gz");
            else if (fname.find(".img") != std::string::npos)
                fname.replace(fname.find(".img"), 4, "_backward.img");
            else if (fname.find(".png") != std::string::npos)
                fname.replace(fname.find(".png"), 4, "_backward.png");
            else if (fname.find(".nrrd") != std::string::npos)
                fname.replace(fname.find(".nrrd"), 5, "_backward.nrrd");
            else fname.append("_backward.nii");
            reg_io_WriteImageFile(outputWarpedImages[1], fname.c_str());
        }
    }
    reg_io_WriteImageFile(outputWarpedImages[0], outputWarpedImageName);

    time_t end;
    time(&end);
    const int minutes = Floor((end - start) / 60.0f);
    const int seconds = static_cast<int>(end - start) - 60 * minutes;
    NR_VERBOSE_APP("Registration performed in " << minutes << " min " << seconds << " sec");
    NR_VERBOSE_APP("Have a good day!");

    return EXIT_SUCCESS;
}
