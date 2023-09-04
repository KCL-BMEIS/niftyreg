/**
 * @file reg_aladin.cpp
 * @author Marc Modat, David C Cash and Pankaj Daga
 * @date 12/08/2009
 *
 * Copyright (c) 2009-2018, University College London
 * Copyright (c) 2018, NiftyReg Developers.
   All rights reserved.
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"
#include "reg_aladin.h"
// #include <libgen.h> //DO NOT WORK ON WINDOWS !

#ifdef _WIN32
#   include <time.h>
#endif

using PrecisionType = float;

void PetitUsage(char *exec) {
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("reg_aladin");
    NR_INFO("Usage:\t" << exec << " -ref <referenceImageName> -flo <floatingImageName> [OPTIONS]");
    NR_INFO("\tSee the help for more details (-h).");
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
}

void Usage(char *exec) {
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Block Matching algorithm for global registration.");
    NR_INFO("Based on Modat et al., \"Global image registration using a symmetric block-matching approach\"");
    NR_INFO("J. Med. Img. 1(2) 024003, 2014, doi: 10.1117/1.JMI.1.2.024003");
    NR_INFO("For any comment, please contact Marc Modat (m.modat@ucl.ac.uk)");
    NR_INFO("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *");
    NR_INFO("Usage:\t" << exec << " -ref <filename> -flo <filename> [OPTIONS]");
    NR_INFO("\t-ref <filename>\tReference image filename (also called Target or Fixed) (mandatory)");
    NR_INFO("\t-flo <filename>\tFloating image filename (also called Source or Moving) (mandatory)");
    NR_INFO("");
    NR_INFO("* * OPTIONS * *");
    NR_INFO("\t-noSym \t\t\tThe symmetric version of the algorithm is used by default. Use this flag to disable it.");
    NR_INFO("\t-rigOnly\t\tTo perform a rigid registration only. (Rigid+affine by default)");
    NR_INFO("\t-affDirect\t\tDirectly optimize 12 DoF affine. (Default is rigid initially then affine)");

    NR_INFO("\t-aff <filename>\t\tFilename which contains the output affine transformation. [outputAffine.txt]");
    NR_INFO("\t-inaff <filename>\tFilename which contains an input affine transformation. (Affine*Reference=Floating) [none]");

    NR_INFO("\t-rmask <filename>\tFilename of a mask image in the reference space.");
    NR_INFO("\t-fmask <filename>\tFilename of a mask image in the floating space. (Only used when symmetric turned on)");
    NR_INFO("\t-res <filename>\t\tFilename of the resampled image. [outputResult.nii.gz]");

    NR_INFO("\t-maxit <int>\t\tMaximal number of iterations of the trimmed least square approach to perform per level. [5]");
    NR_INFO("\t-ln <int>\t\tNumber of levels to use to generate the pyramids for the coarse-to-fine approach. [3]");
    NR_INFO("\t-lp <int>\t\tNumber of levels to use to run the registration once the pyramids have been created. [ln]");

    NR_INFO("\t-smooR <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Reference image. [0]");
    NR_INFO("\t-smooF <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Floating image. [0]");
    NR_INFO("\t-refLowThr <float>\tLower threshold value applied to the reference image. [0]");
    NR_INFO("\t-refUpThr <float>\tUpper threshold value applied to the reference image. [0]");
    NR_INFO("\t-floLowThr <float>\tLower threshold value applied to the floating image. [0]");
    NR_INFO("\t-floUpThr <float>\tUpper threshold value applied to the floating image. [0]");
    NR_INFO("\t-pad <float>\t\tPadding value [nan]");

    NR_INFO("\t-nac\t\t\tUse the nifti header origin to initialise the transformation. (Image centres are used by default)");
    NR_INFO("\t-comm\t\t\tUse the input masks centre of mass to initialise the transformation. (Image centres are used by default)");
    NR_INFO("\t-comi\t\t\tUse the input images centre of mass to initialise the transformation. (Image centres are used by default)");
    NR_INFO("\t-interp\t\t\tInterpolation order to use internally to warp the floating image.");
    NR_INFO("\t-iso\t\t\tMake floating and reference images isotropic if required.");

    NR_INFO("\t-pv <int>\t\tPercentage of blocks to use in the optimisation scheme. [50]");
    NR_INFO("\t-pi <int>\t\tPercentage of blocks to consider as inlier in the optimisation scheme. [50]");
    NR_INFO("\t-speeeeed\t\tGo faster");

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

    //   NR_INFO("\t-crv\t\t\tChoose custom capture range for the block matching alg");
#ifdef _OPENMP
    int defaultOpenMPValue = omp_get_num_procs();
    if (getenv("OMP_NUM_THREADS") != nullptr)
        defaultOpenMPValue = atoi(getenv("OMP_NUM_THREADS"));
    NR_INFO("\t-omp <int>\t\tNumber of threads to use with OpenMP. [" << defaultOpenMPValue << "/" << omp_get_num_procs() << "]");
#endif
    NR_INFO("\t-voff\t\t\tTurns verbose off [on]");
    NR_INFO("");
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

    int symFlag = 1;

    char *referenceImageName = nullptr;
    int referenceImageFlag = 0;

    char *floatingImageName = nullptr;
    int floatingImageFlag = 0;

    const char *outputAffineName = "outputAffine.txt";
    int outputAffineFlag = 0;

    char *inputAffineName = nullptr;
    int inputAffineFlag = 0;

    char *referenceMaskName = nullptr;
    int referenceMaskFlag = 0;

    char *floatingMaskName = nullptr;
    int floatingMaskFlag = 0;

    const char *outputResultName = "outputResult.nii.gz";
    int outputResultFlag = 0;

    int maxIter = 5;
    int nLevels = 3;
    int levelsToPerform = std::numeric_limits<int>::max();
    int affineFlag = 1;
    int rigidFlag = 1;
    int blockStepSize = 1;
    int blockPercentage = 50;
    int inlierLts = 50;
    int alignCentre = 1;
    int alignCentreOfMass = 0;
    int interpolation = 1;
    float floatingSigma = 0;
    float referenceSigma = 0;

    float referenceLowerThr = std::numeric_limits<PrecisionType>::lowest();
    float referenceUpperThr = std::numeric_limits<PrecisionType>::max();
    float floatingLowerThr = std::numeric_limits<PrecisionType>::lowest();
    float floatingUpperThr = std::numeric_limits<PrecisionType>::max();
    float paddingValue = std::numeric_limits<PrecisionType>::quiet_NaN();

    bool iso = false;
    bool verbose = true;
    int captureRangeVox = 3;
    PlatformType platformType(PlatformType::Cpu);
    unsigned gpuIdx = 999;

#ifdef _OPENMP
    // Set the default number of threads
    int defaultOpenMPValue = omp_get_num_procs();
    if (getenv("OMP_NUM_THREADS") != nullptr)
        defaultOpenMPValue = atoi(getenv("OMP_NUM_THREADS"));
    omp_set_num_threads(defaultOpenMPValue);
#endif

    /* read the input parameter */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-Help") == 0 ||
            strcmp(argv[i], "-HELP") == 0 || strcmp(argv[i], "-h") == 0 ||
            strcmp(argv[i], "--h") == 0 || strcmp(argv[i], "--help") == 0) {
            Usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "--xml") == 0) {
            NR_COUT << xml_aladin;
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "-version") == 0 ||
            strcmp(argv[i], "-Version") == 0 ||
            strcmp(argv[i], "-V") == 0 ||
            strcmp(argv[i], "-v") == 0 ||
            strcmp(argv[i], "--v") == 0 ||
            strcmp(argv[i], "--version") == 0) {
            NR_COUT << NR_VERSION << std::endl;
            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "-ref") == 0 || strcmp(argv[i], "-target") == 0 || strcmp(argv[i], "--ref") == 0) {
            referenceImageName = argv[++i];
            referenceImageFlag = 1;
        } else if (strcmp(argv[i], "-flo") == 0 || strcmp(argv[i], "-source") == 0 || strcmp(argv[i], "--flo") == 0) {
            floatingImageName = argv[++i];
            floatingImageFlag = 1;
        }

        else if (strcmp(argv[i], "-noSym") == 0 || strcmp(argv[i], "--noSym") == 0) {
            symFlag = 0;
        } else if (strcmp(argv[i], "-aff") == 0 || strcmp(argv[i], "--aff") == 0) {
            outputAffineName = argv[++i];
            outputAffineFlag = 1;
        } else if (strcmp(argv[i], "-inaff") == 0 || strcmp(argv[i], "--inaff") == 0) {
            inputAffineName = argv[++i];
            inputAffineFlag = 1;
        } else if (strcmp(argv[i], "-rmask") == 0 || strcmp(argv[i], "-tmask") == 0 || strcmp(argv[i], "--rmask") == 0) {
            referenceMaskName = argv[++i];
            referenceMaskFlag = 1;
        } else if (strcmp(argv[i], "-fmask") == 0 || strcmp(argv[i], "-smask") == 0 || strcmp(argv[i], "--fmask") == 0) {
            floatingMaskName = argv[++i];
            floatingMaskFlag = 1;
        } else if (strcmp(argv[i], "-res") == 0 || strcmp(argv[i], "-result") == 0 || strcmp(argv[i], "--res") == 0) {
            outputResultName = argv[++i];
            outputResultFlag = 1;
        } else if (strcmp(argv[i], "-maxit") == 0 || strcmp(argv[i], "--maxit") == 0) {
            maxIter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ln") == 0 || strcmp(argv[i], "--ln") == 0) {
            nLevels = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lp") == 0 || strcmp(argv[i], "--lp") == 0) {
            levelsToPerform = atoi(argv[++i]);
        }

        else if (strcmp(argv[i], "-smooR") == 0 || strcmp(argv[i], "-smooT") == 0 || strcmp(argv[i], "--smooR") == 0) {
            referenceSigma = (float)(atof(argv[++i]));
        } else if (strcmp(argv[i], "-smooF") == 0 || strcmp(argv[i], "-smooS") == 0 || strcmp(argv[i], "--smooF") == 0) {
            floatingSigma = (float)(atof(argv[++i]));
        } else if (strcmp(argv[i], "-rigOnly") == 0 || strcmp(argv[i], "--rigOnly") == 0) {
            rigidFlag = 1;
            affineFlag = 0;
        } else if (strcmp(argv[i], "-affDirect") == 0 || strcmp(argv[i], "--affDirect") == 0) {
            rigidFlag = 0;
            affineFlag = 1;
        } else if (strcmp(argv[i], "-nac") == 0 || strcmp(argv[i], "--nac") == 0) {
            alignCentre = 0;
        } else if (strcmp(argv[i], "-comm") == 0 || strcmp(argv[i], "--comm") == 0 ||
                  strcmp(argv[i], "-cog") == 0 || strcmp(argv[i], "--cog") == 0) {
            alignCentre = 0;
            alignCentreOfMass = 1;
        } else if (strcmp(argv[i], "-comi") == 0 || strcmp(argv[i], "--comi") == 0) {
            alignCentre = 0;
            alignCentreOfMass = 2;
        } else if (strcmp(argv[i], "-%v") == 0 || strcmp(argv[i], "-pv") == 0 || strcmp(argv[i], "--pv") == 0) {
            int value = atoi(argv[++i]);
            if (value < 1 || value > 100) {
                NR_ERROR("The variance argument is expected to be an integer between 1 and 100");
                return EXIT_FAILURE;
            }
            blockPercentage = value;
        } else if (strcmp(argv[i], "-%i") == 0 || strcmp(argv[i], "-pi") == 0 || strcmp(argv[i], "--pi") == 0) {
            int value = atoi(argv[++i]);
            if (value < 1 || value > 100) {
                NR_ERROR("The inlier argument is expected to be an integer between 1 and 100");
                return EXIT_FAILURE;
            }
            inlierLts = value;
        } else if (strcmp(argv[i], "-speeeeed") == 0 || strcmp(argv[i], "--speeed") == 0) {
            blockStepSize = 2;
        } else if (strcmp(argv[i], "-interp") == 0 || strcmp(argv[i], "--interp") == 0) {
            interpolation = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-refLowThr") == 0 || strcmp(argv[i], "--refLowThr") == 0) {
            referenceLowerThr = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-refUpThr") == 0 || strcmp(argv[i], "--refUpThr") == 0) {
            referenceUpperThr = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-floLowThr") == 0 || strcmp(argv[i], "--floLowThr") == 0) {
            floatingLowerThr = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-floUpThr") == 0 || strcmp(argv[i], "--floUpThr") == 0) {
            floatingUpperThr = std::stof(argv[++i]);
        }

        else if (strcmp(argv[i], "-pad") == 0 || strcmp(argv[i], "--pad") == 0) {
            paddingValue = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-iso") == 0 || strcmp(argv[i], "--iso") == 0) {
            iso = true;
        } else if (strcmp(argv[i], "-voff") == 0 || strcmp(argv[i], "--voff") == 0) {
            NR_DEBUG("The verbose cannot be switch off in debug");
#ifdef NDEBUG
            verbose = false;
#endif
        } else if (strcmp(argv[i], "-platf") == 0 || strcmp(argv[i], "--platf") == 0) {
            PlatformType value{ atoi(argv[++i]) };
            if (value < PlatformType::Cpu || value > PlatformType::OpenCl) {
                NR_ERROR("The platform argument is expected to be 0, 1 or 2 | 0=CPU, 1=CUDA 2=OPENCL");
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
        } else if (strcmp(argv[i], "-crv") == 0 || strcmp(argv[i], "--crv") == 0) {
            captureRangeVox = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-omp") == 0 || strcmp(argv[i], "--omp") == 0) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            NR_WARN("NiftyReg has not been compiled with OpenMP, the \'-omp\' flag is ignored");
            ++i;
#endif
        } else {
            NR_ERROR("\tParameter " << argv[i] << " unknown!");
            PetitUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (!referenceImageFlag || !floatingImageFlag) {
        NR_ERROR("The reference and the floating image have to be defined!");
        PetitUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Output the command line
    PrintCmdLine(argc, argv, verbose);

    unique_ptr<reg_aladin<PrecisionType>> reg;
    if (symFlag) {
        reg.reset(new reg_aladin_sym<PrecisionType>);
        if ((referenceMaskFlag && !floatingMaskName) || (!referenceMaskFlag && floatingMaskName)) {
            NR_WARN("You have one image mask option turned on but not the other.");
            NR_WARN("This will affect the degree of symmetry achieved.");
        }
    } else {
        reg.reset(new reg_aladin<PrecisionType>);
        if (floatingMaskFlag) {
            NR_WARN("Note: Floating mask flag only used in symmetric method. Ignoring this option");
        }
    }

    /* Read the reference image and check its dimension */
    NiftiImage referenceHeader = reg_io_ReadImageFile(referenceImageName);
    if (!referenceHeader) {
        NR_ERROR("Error when reading the reference image: " << referenceImageName);
        return EXIT_FAILURE;
    }

    /* Read the floating image and check its dimension */
    NiftiImage floatingHeader = reg_io_ReadImageFile(floatingImageName);
    if (!floatingHeader) {
        NR_ERROR("Error when reading the floating image: " << floatingImageName);
        return EXIT_FAILURE;
    }

    // Set the reference and floating images
    // make the images isotropic if required
    reg->SetInputReference(iso ? NiftiImage(reg_makeIsotropic(referenceHeader, 1)) : referenceHeader);
    reg->SetInputFloating(iso ? NiftiImage(reg_makeIsotropic(floatingHeader, 1)) : floatingHeader);

    /* read the reference mask image */
    if (referenceMaskFlag) {
        NiftiImage referenceMaskImage = reg_io_ReadImageFile(referenceMaskName);
        if (!referenceMaskImage) {
            NR_ERROR("Error when reading the reference mask image: " << referenceMaskName);
            return EXIT_FAILURE;
        }
        /* check the dimension */
        for (int i = 1; i <= referenceHeader->dim[0]; i++) {
            if (referenceHeader->dim[i] != referenceMaskImage->dim[i]) {
                NR_ERROR("The reference image and its mask do not have the same dimension");
                return EXIT_FAILURE;
            }
        }
        // make the image isotropic if required
        reg->SetInputMask(iso ? NiftiImage(reg_makeIsotropic(referenceMaskImage, 0)) : std::move(referenceMaskImage));
    }
    /* Read the floating mask image */
    if (floatingMaskFlag && symFlag) {
        NiftiImage floatingMaskImage = reg_io_ReadImageFile(floatingMaskName);
        if (!floatingMaskImage) {
            NR_ERROR("Error when reading the floating mask image: " << floatingMaskName);
            return EXIT_FAILURE;
        }
        /* check the dimension */
        for (int i = 1; i <= floatingHeader->dim[0]; i++) {
            if (floatingHeader->dim[i] != floatingMaskImage->dim[i]) {
                NR_ERROR("The floating image and its mask do not have the same dimension");
                return EXIT_FAILURE;
            }
        }
        // make the image isotropic if required
        reg->SetInputFloatingMask(iso ? NiftiImage(reg_makeIsotropic(floatingMaskImage, 0)) : std::move(floatingMaskImage));
    }

    reg->SetMaxIterations(maxIter);
    reg->SetNumberOfLevels(nLevels);
    reg->SetLevelsToPerform(levelsToPerform);
    reg->SetReferenceSigma(referenceSigma);
    reg->SetFloatingSigma(floatingSigma);
    reg->SetAlignCentre(alignCentre);
    reg->SetAlignCentreMass(alignCentreOfMass);
    reg->SetPerformAffine(affineFlag);
    reg->SetPerformRigid(rigidFlag);
    reg->SetBlockStepSize(blockStepSize);
    reg->SetBlockPercentage(blockPercentage);
    reg->SetInlierLts(inlierLts);
    reg->SetInterpolation(interpolation);
    reg->SetCaptureRangeVox(captureRangeVox);
    reg->SetPlatformType(platformType);
    reg->SetGpuIdx(gpuIdx);

    if (referenceLowerThr != referenceUpperThr) {
        reg->SetReferenceLowerThreshold(referenceLowerThr);
        reg->SetReferenceUpperThreshold(referenceUpperThr);
    }

    if (floatingLowerThr != floatingUpperThr) {
        reg->SetFloatingLowerThreshold(floatingLowerThr);
        reg->SetFloatingUpperThreshold(floatingUpperThr);
    }

    reg->SetWarpedPaddingValue(paddingValue);

    if (reg->GetLevelsToPerform() > reg->GetNumberOfLevels())
        reg->SetLevelsToPerform(reg->GetNumberOfLevels());

    // Set the input affine transformation if defined
    if (inputAffineFlag == 1)
        reg->SetInputTransform(inputAffineName);

    // Set the verbose type
    reg->SetVerbose(verbose);

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

    // The warped image is saved
    if (iso) {
        reg->SetInputReference(referenceHeader);
        reg->SetInputFloating(floatingHeader);
    }
    NiftiImage outputResultImage = reg->GetFinalWarpedImage();
    reg_io_WriteImageFile(outputResultImage, outputResultName);

    /* The affine transformation is saved */
    reg_tool_WriteAffineFile(reg->GetTransformationMatrix(), outputAffineName);

    time_t end;
    time(&end);
    const int minutes = Floor((end - start) / 60.0f);
    const int seconds = static_cast<int>(end - start) - 60 * minutes;
    NR_VERBOSE_APP("Registration performed in " << minutes << " min " << seconds << " sec");
    NR_VERBOSE_APP("Have a good day!");

    return EXIT_SUCCESS;
}
