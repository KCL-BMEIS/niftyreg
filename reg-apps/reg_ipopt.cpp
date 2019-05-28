/*
 *  reg_ipopt.cpp
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_f3d2_ipopt.h"
#include "_reg_f3d_ipopt.h"
#include "IpIpoptApplication.hpp"
#include "command_line_reader_reg_ipopt.h"
#include "exception.h"
#include <float.h>
#include <cstdio>
#include <sys/stat.h>
//#include <boost/filesystem.hpp>

using namespace Ipopt;

void clip_last_percentile(nifti_image *img) {
  // make sure the img datatype is float
  reg_tools_changeDatatype<float>(img);
  // Create a copy of the reference image to extract the robust range
  nifti_image *temp_reference = nifti_copy_nim_info(img);
  temp_reference->data = (void *)malloc(temp_reference->nvox * temp_reference->nbyper);
  memcpy(temp_reference->data, img->data, temp_reference->nvox * temp_reference->nbyper);
//  reg_tools_changeDatatype<float>(temp_reference);
  // Extract the robust range of the reference image
  auto *refDataPtr = static_cast<float *>(temp_reference->data);
  reg_heapSort(refDataPtr, temp_reference->nvox);
  // get the last percentile
  float perc = refDataPtr[(int) std::round((float) temp_reference->nvox * 0.99f)];
  float min = refDataPtr[0];
#ifndef NDEBUG
  std::cout << "99% percentile = " << perc << std::endl;
  std::cout << "min intensity = " << min << std::endl;
#endif
  // free the temporary image
  nifti_image_free(temp_reference);
  // clip the input image intensity values
  auto *imgPtr = static_cast<float *>(img->data);
  for (int i=0; i < img->nvox; ++i) {
    if(imgPtr[i] > perc) {
      imgPtr[i] = perc;
    }
  }

}

int main(int argc, char** argv) {
  // Set time variables for logs
  time_t start;
  time_t startLevel;
  time_t end;
  time(&start);

 // Read the command line options
  CommandLineReaderRegIpopt::getInstance().processCmdLineOptions(argc, argv);

  // If the user asks for help print help and close program
  if (CommandLineReaderRegIpopt::getInstance().justHelp()) {
    CommandLineReaderRegIpopt::getInstance().printUsage(std::cout);
    return EXIT_SUCCESS;
  }

  // create the directory (with commonly used permissions)
  std::string saveDir = CommandLineReaderRegIpopt::getInstance().getOutDir();
  const int check = mkdir(saveDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  // write the command line that was used
  CommandLineReaderRegIpopt::getInstance().writeCommandLine(argc, argv);

  // Read the reference and floating images, constraint mask and init cpp
  nifti_image *referenceImage = NULL;
  nifti_image *floatingImage = NULL;
  nifti_image *maskImage = NULL;
  nifti_image *initCPP = NULL;

  referenceImage = reg_io_ReadImageFile(CommandLineReaderRegIpopt::getInstance().getRefFilePath().c_str());
  if (!referenceImage) {
    throw CouldNotReadInputImage(CommandLineReaderRegIpopt::getInstance().getRefFilePath());
  }

  floatingImage = reg_io_ReadImageFile(CommandLineReaderRegIpopt::getInstance().getFloFilePath().c_str());
  if (!floatingImage) {
    throw CouldNotReadInputImage(CommandLineReaderRegIpopt::getInstance().getFloFilePath());
  }

  // Normalisation
//  clip_last_percentile(referenceImage);
//  clip_last_percentile(floatingImage);

  std::string initCPPPath = CommandLineReaderRegIpopt::getInstance().getInitCPPPath();
  if (initCPPPath.length() > 1) {
      initCPP = reg_io_ReadImageFile(initCPPPath.c_str());
  }

 // Create the registration object
  SmartPtr<reg_f3d2_ipopt<double> > REG = new reg_f3d2_ipopt<double>(referenceImage->nt, floatingImage->nt);
//  SmartPtr<reg_f3d_ipopt<double> > REG = new reg_f3d_ipopt<double>(referenceImage->nt, floatingImage->nt);
  REG->SetReferenceImage(referenceImage);
  REG->SetFloatingImage(floatingImage);

  // add mask constraint if constraints are used
  if (CommandLineReaderRegIpopt::getInstance().getUseConstraint()) {
      std::string maskPath = CommandLineReaderRegIpopt::getInstance().getMaskFilePath();
      if (maskPath.length() > 1) {
          std::cout << "Load incompressibility mask from "<< maskPath << std::endl;
          maskImage = reg_io_ReadImageFile(maskPath.c_str());
          REG->setConstraintMask(maskImage);
      }
      else {  // create a mask that covers all the image space
          std::cout << "Incompressibility constraint is imposed everywhere in the spatial domain" << std::endl;
          REG->setFullConstraint();
      }
  }
//  if (maskImage != NULL) {
//      REG->setConstraintMask(maskImage);
//  }
  if (initCPP != NULL) {
      REG->SetControlPointGridImage(initCPP);
  }
  REG->setSaveDir(saveDir);

  REG->setBSplineType(CommandLineReaderRegIpopt::getInstance().getBSplineType());

  REG->setDivergenceConstraint(CommandLineReaderRegIpopt::getInstance().getUseConstraint());

//  REG->setSaveMoreOutput(CommandLineReaderRegIpopt::getInstance().getSaveMoreOutput());

//  REG->SetWarpedPaddingValue(0.);

  // interpolation of the images (change the parameter ref_f3d::interpolation)
  // (note that the control point grids are always parameterised by cubic B-splines)
//  REG->UseCubicSplineInterpolation();
  REG->UseLinearInterpolation();

  // Gradient options
//  REG->UseApproximatedGradient();
  REG->DoNotUseConjugateGradient();
  REG->SetGradientSmoothingSigma(0.);

  // Set the objective function (default is NMI)
//  bool normalise = false;
//  REG->UseSSD(0, normalise);
  float spacing_mm = 5.f;
  REG->SetSpacing(0, spacing_mm);  // do not use a spacing of 5 to avoid lut...
  std::cout << "Set spacing to " << spacing_mm << " mm" << std::endl;
  REG->SetLinearEnergyWeight(0.f);  // default is 0.01
  REG->SetBendingEnergyWeight(0.05f);  // default is 0.001 and 0.1 works ok for SSFP
  REG->SetInverseConsistencyWeight(0.f);  // make sure inverse consistency is not used
//  float scale = 1.f;  // scale for SSD
//  float scale = 1e7;  // appropriate scaling factor for NMI
  float scale = 100000.f;  // appropriate scaling factor for LNCC
//  REG->UseLNCC(0, 2.f);
  REG->setScale(scale);

//  int maxIter = 5;
  int maxIter = 200;

  // Set the number of levels to perform for the pyramidal approach
  unsigned int levelToPerform = CommandLineReaderRegIpopt::getInstance().getLevelToPerform();
  if (levelToPerform <= 1){
    REG->DoNotUsePyramidalApproach();
  }
  // number of level to perform starting from the one with the lowest resolution
  REG->SetLevelToPerform(levelToPerform);
  // number of downsampling to perform
  REG->SetLevelNumber(levelToPerform);

  REG->printConfigInfo();

#if defined (_OPENMP)
  int maxThreadNumber = omp_get_max_threads();
//  int maxThreadNumber = omp_get_max_threads();
  std::cout << "OpenMP is used with " << maxThreadNumber << " thread(s)." << std::endl;
#endif // _OPENMP

//  ApplicationReturnStatus status;
//  SmartPtr<TNLP> mynlp = REG;
//  SmartPtr<IpoptApplication> app;

  for(int level=0; level<levelToPerform; level++) {
//    std::cout << "ready to start" << std::endl;
    time(&startLevel);
    // all NiftyReg variables are initialised for the current level
    REG->initLevel(level);
    // gradient check
//    REG->gradientCheck();
//    REG->voxelBasedGradientCheck();
//    REG->printImgStat();
//    REG->CheckWarpImageGradient();
//    reg_exit();

    SmartPtr<TNLP> mynlp = REG;
    ApplicationReturnStatus status;

    // Create a new instance of IpoptApplication
    SmartPtr<IpoptApplication> app = new IpoptApplication();
//    app = new IpoptApplication();

    // Set IpoptApplication options
    app->Options()->SetStringValue("jac_c_constant", "yes");  // all constraints are linear
    app->Options()->SetStringValue("jac_d_constant", "yes");
    // Quasi-Newton options
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetIntegerValue("limited_memory_max_history", 12);  // default is 6
    // linear solver options
//    app->Options()->SetStringValue("nlp_scaling_method", "equilibration-based");
//    app->Options()->SetStringValue("linear_solver", "pardiso");  // ma27 or ma86
    app->Options()->SetStringValue("linear_solver", "ma57");  // ma27 or ma86
    // ma57 options
      app->Options()->SetStringValue("ma57_automatic_scaling", "no");
    // ma27 options
//    app->Options()->SetStringValue("linear_system_scaling", "mc19");
//    app->Options()->SetStringValue("linear_scaling_on_demand", "no");
    // ma86 options
//    app->Options()->SetIntegerValue("ma86_print_level", 1);  // default -1
//    app->Options()->SetStringValue("ma86_scaling", "mc77");
    // more options
//    app->Options()->SetStringValue("neg_curv_test_reg", "no");  // no -> original IPOPT; default yes
    app->Options()->SetStringValue("print_timing_statistics", "yes");
    app->Options()->SetStringValue("accept_every_trial_step", "no");  // if "yes", deactivate line search
    app->Options()->SetIntegerValue("print_level", 5);  // between 1 and 12

    // set options for the convergence criteria
    app->Options()->SetNumericValue("tol", scale*1e-6);  // default scale*1e-4
    app->Options()->SetNumericValue("acceptable_obj_change_tol", 1e-5);  // stop criteria based on objective
    app->Options()->SetNumericValue("acceptable_tol", scale*100);  // default scale*1e-3
    app->Options()->SetNumericValue("acceptable_compl_inf_tol", 10000.);  // default 0.01
    app->Options()->SetIntegerValue("acceptable_iter", 15);  // default 15
    if (level == levelToPerform - 1){
      app->Options()->SetIntegerValue("max_iter", maxIter);
      // save more output for the last level if asked by the user
      REG->setSaveMoreOutput(CommandLineReaderRegIpopt::getInstance().getSaveMoreOutput());
    }
    else {
      app->Options()->SetIntegerValue("max_iter", 100*(levelToPerform - level));
//      app->Options()->SetIntegerValue("max_iter", 300);
    }
      app->Options()->SetStringValue("print_info_string", "yes");  // for more info at each iter


    // Intialize the IpoptApplication and process the options
    app->Initialize();

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(mynlp);

    REG->clearLevel(level);

    time(&end);
    int minutes=(int)floorf((end-startLevel)/60.0f);
    int seconds=(int)(end-startLevel - 60*minutes);
    std::string text = stringFormat("Registration level %d performed in %i min %i sec",
            level+1, minutes, seconds);
    reg_print_info((argv[0]), text.c_str());

    if (level == levelToPerform - 1) {
      // print status for the final step
      switch (status) {
        case Solve_Succeeded:
          std::cout << std::endl << "*** Optimal solution found" << std::endl;
              break;
        case Solved_To_Acceptable_Level:
          std::cout << std::endl << "*** Acceptable solution found" << std::endl;
              break;
        case Maximum_Iterations_Exceeded:
          std::cout << std::endl
                    << "*** Return current best solution after reaching the maximum number of iterations" << std::endl;
              break;
        default:
          std::cout << std::endl << "*** The problem FAILED" << std::endl;
              break;
      }
    }
  }
  // print stats info
  REG->PrintStatInfo();

  // Print total time for the registration
  time(&end);
  int minutes=(int)floorf((end-start)/60.0f);
  int seconds=(int)(end-start - 60*minutes);
  std::cout << std::endl;
  std::string text = stringFormat("Registration performed in %i min %i sec", minutes, seconds);
  reg_print_info((argv[0]), text.c_str());
  reg_print_info((argv[0]), "Have a good day !");

  return 0;
}
