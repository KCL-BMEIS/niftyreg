/*
 *  reg_ipopt.cpp
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_f3d2_ipopt.h"
#include "_reg_f3d_ipopt.h"
#include "IpIpoptApplication.hpp"
#include "command_line_reader.h"
#include "exception.h"
#include <float.h>
#include <cstdio>
#include <sys/stat.h>
//#include <boost/filesystem.hpp>

using namespace Ipopt;

int main(int argc, char** argv){
  // Set time variables for logs
  time_t start;
  time_t startLevel;
  time_t end;
  time(&start);

 // Read the command line options
  CommandLineReader::getInstance().processCmdLineOptions(argc, argv);

  // If the user asks for help print help and close program
  if (CommandLineReader::getInstance().justHelp()) {
    CommandLineReader::getInstance().printUsage(std::cout);
    return EXIT_SUCCESS;
  }

  // create the directory (with commonly used permissions)
  std::string saveDir = CommandLineReader::getInstance().getOutDir();
  const int check = mkdir(saveDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

//  boost::filesystem::path saveDir(CommandLineReader::getInstance().getOutDir().c_str());
//  if (!boost::filesystem::exists(saveDir)) {
//      boost::filesystem::create_directory(saveDir);
//  }

  // write the command line that was used
  CommandLineReader::getInstance().writeCommandLine(argc, argv);

  // Read the reference and floating image
  nifti_image *referenceImage = NULL;
  nifti_image *floatingImage = NULL;
  nifti_image *maskImage = NULL;

  referenceImage = reg_io_ReadImageFile(CommandLineReader::getInstance().getRefFilePath().c_str());
  if (!referenceImage) {
    throw CouldNotReadInputImage(CommandLineReader::getInstance().getRefFilePath());
  }

  floatingImage = reg_io_ReadImageFile(CommandLineReader::getInstance().getFloFilePath().c_str());
  if (!floatingImage) {
    throw CouldNotReadInputImage(CommandLineReader::getInstance().getFloFilePath());
  }

  std::string maskPath = CommandLineReader::getInstance().getMaskFilePath();
  if (maskPath.length() > 1) {
    maskImage = reg_io_ReadImageFile(maskPath.c_str());
  }

  // Normalize data
//  float scaleIntensity = 100.f;
//  float meanRef = reg_tools_getMeanValue(referenceImage);
//  reg_tools_substractValueToImage(referenceImage, referenceImage, meanRef);
//  float stdRef = reg_tools_getSTDValue(referenceImage);
//  reg_tools_divideValueToImage(referenceImage, referenceImage, stdRef/scaleIntensity);
//  float meanFlo = reg_tools_getMeanValue(floatingImage);
//  reg_tools_substractValueToImage(floatingImage, floatingImage, meanFlo);
//  float stdFlo = reg_tools_getSTDValue(floatingImage);
//  reg_tools_divideValueToImage(floatingImage, floatingImage, stdFlo/scaleIntensity);

 // Create the registration object
//  SmartPtr<reg_f3d2_ipopt<float> > REG = new reg_f3d2_ipopt<float>(referenceImage->nt, floatingImage->nt);
  SmartPtr<reg_f3d2_ipopt<double> > REG = new reg_f3d2_ipopt<double>(referenceImage->nt, floatingImage->nt);
//  SmartPtr<reg_f3d_ipopt<float> > REG = new reg_f3d_ipopt<float>(referenceImage->nt, floatingImage->nt);
//  SmartPtr<reg_f3d_ipopt<double> > REG = new reg_f3d_ipopt<double>(referenceImage->nt, floatingImage->nt);
  REG->SetReferenceImage(referenceImage);
  REG->SetFloatingImage(floatingImage);
  if (maskImage != NULL) {
      REG->setConstraintMask(maskImage);
  }
  REG->setSaveDir(saveDir);

  REG->setDivergenceConstraint(CommandLineReader::getInstance().getUseConstraint());
//  REG->setDivergenceConstraint(true);

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
  REG->SetSpacing(0, 5.f);  // do not use a spacing of 5 to avoid lut...
  std::cout << "Set spacing to " << 5 << std::endl;
//  REG->SetLinearEnergyWeight(0.f);  // default is 0.01
//  REG->SetBendingEnergyWeight(0.f);  // default is 0.001
  REG->SetInverseConsistencyWeight(0.f);  // make sure inverse consistency is not used
  float scale = 1e7;
  REG->setScale(scale);  // appropriate scaling factor for NMI

//  int maxIter = 1;
  int maxIter = 300;

  // Set the number of levels to perform for the pyramidal approach
  unsigned int levelToPerform = 1;
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
  std::cout << "OpenMP is used with " << maxThreadNumber << " thread(s)." << std::endl;
#endif // _OPENMP

  ApplicationReturnStatus status;

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

    // Create a new instance of IpoptApplication
    SmartPtr<IpoptApplication> app = new IpoptApplication();

    // Set IpoptApplication options
    app->Options()->SetStringValue("jac_c_constant", "yes");  // all constraints are linear
    app->Options()->SetStringValue("jac_d_constant", "yes");
    // Quasi-Newton options
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetIntegerValue("limited_memory_max_history", 12);  // default is 6
    // linear solver options
//    app->Options()->SetStringValue("nlp_scaling_method", "equilibration-based");
    app->Options()->SetStringValue("linear_solver", "ma86");  // ma27 or ma86
//    app->Options()->SetStringValue("linear_solver", "ma27");  // ma27 or ma86
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
    if (level == levelToPerform - 1){
      app->Options()->SetNumericValue("tol", scale*1e-4);
      app->Options()->SetNumericValue("acceptable_obj_change_tol", 1e-5);  // stop criteria based on objective
      app->Options()->SetNumericValue("acceptable_tol", scale*1e-3);
      app->Options()->SetIntegerValue("acceptable_iter", 15);  // default 15
      app->Options()->SetIntegerValue("max_iter", maxIter);
//      app->Options()->SetIntegerValue("max_iter", 150);
    }
    else {
      app->Options()->SetNumericValue("tol", 1e-6);
      app->Options()->SetIntegerValue("max_iter", 1);
//      app->Options()->SetIntegerValue("max_iter", 300);
    }
      app->Options()->SetStringValue("print_info_string", "yes");  // for more info at each iter
    //  app->Options()->SetStringValue("jac_c_constant", "yes");

    // TODO: Gradient check will mess up your controlPointGrid and so the initialisation...
//    app->Options()->SetStringValue("derivative_test", "first-order");
//    app->Options()->SetStringValue("derivative_test_print_all", "yes");
    // in reg_f3d in ApproximatedGradient eps = this->controlPointGrid->dx / 100.f is used
//    app->Options()->SetNumericValue("derivative_test_perturbation", 1e-2);
//    app->Options()->SetIntegerValue("derivative_test_first_index", 20770);
//      app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");

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
  }
  // print stats info
  REG->PrintStatInfo();

  // print status for the final step
  switch (status) {
    case Solve_Succeeded:
      std::cout << std::endl << "*** Optimal solution found" << std::endl;
      break;
    case Solved_To_Acceptable_Level:
      std::cout << std::endl << "*** Acceptable solution found" << std::endl;
      break;
    default:
      std::cout << std::endl << "*** The problem FAILED" << std::endl;
      break;
  }
  // Print total time for the registration
  time(&end);
  int minutes=(int)floorf((end-start)/60.0f);
  int seconds=(int)(end-start - 60*minutes);
  std::cout << std::endl;
  std::string text = stringFormat("Registration performed in %i min %i sec", minutes, seconds);
  reg_print_info((argv[0]), text.c_str());
  reg_print_info((argv[0]), "Have a good day !");

  return (int) status;
}
