/*
 *  reg_ipopt.cpp
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteMatrix.h"
#include "_reg_f3d2_ipopt.h"
//#include "reg_ipopt.h"
#include "IpIpoptApplication.hpp"
#include "command_line_reader.h"
#include "exception.h"
#include <float.h>
#include <cstdio>

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

  // Read the reference and floating image
  nifti_image *referenceImage=NULL;
  nifti_image *floatingImage=NULL;

  referenceImage=reg_io_ReadImageFile(CommandLineReader::getInstance().getRefFilePath().c_str());
  if (!referenceImage){
    throw CouldNotReadInputImage(CommandLineReader::getInstance().getRefFilePath());
  }

  floatingImage=reg_io_ReadImageFile(CommandLineReader::getInstance().getFloFilePath().c_str());
  if (!floatingImage){
    throw CouldNotReadInputImage(CommandLineReader::getInstance().getFloFilePath());
  }

 // Create the registration object
//  SmartPtr<reg_f3d2_ipopt<float> > REG = new reg_f3d2_ipopt<float>(referenceImage->nt, floatingImage->nt);
  SmartPtr<reg_f3d2_ipopt<double> > REG = new reg_f3d2_ipopt<double>(referenceImage->nt, floatingImage->nt);
  REG->SetReferenceImage(referenceImage);
  REG->SetFloatingImage(floatingImage);

  // Set hyperparameters
  REG->UseCubicSplineInterpolation();
//  REG->UseApproximatedGradient();
  REG->DoNotUseConjugateGradient();

  // Set the objective function (default is NMI)
//  bool normalise = false;
//  REG->UseSSD(0, normalise);
//  REG->SetSpacing(0, 10.f);
//  REG->SetLinearEnergyWeight(0.f);
//  REG->SetBendingEnergyWeight(0.001f);
  REG->SetInverseConsistencyWeight(0.f);  // make sure inverse consistency is not used

  // Set the number of levels to perform for the pyramidal approach
  unsigned int levelToPerform = 1;
  if (levelToPerform <= 1){
    REG->DoNotUsePyramidalApproach();
  }
  // number of level to perform starting from the one with the lowest resolution
  REG->SetLevelToPerform(levelToPerform);
  // number of downsampling to perform
  REG->SetLevelNumber(levelToPerform);

  ApplicationReturnStatus status;

  for(int level=0; level<levelToPerform; level++) {
//    std::cout << "ready to start" << std::endl;
    time(&startLevel);
    // all NiftyReg variables are initialised for the current level
    REG->initLevel(level);
    SmartPtr<TNLP> mynlp = REG;

    // Create a new instance of IpoptApplication
    SmartPtr<IpoptApplication> app = new IpoptApplication();

    // Set IpoptApplication options
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetIntegerValue("print_level", 5);  // between 1 and 12
    if (level == levelToPerform - 1){
      app->Options()->SetNumericValue("tol", 1e-6);
      app->Options()->SetIntegerValue("max_iter", 150);
    }
    else {
      app->Options()->SetNumericValue("tol", 1e-6);
      app->Options()->SetIntegerValue("max_iter", 300);
    }
    //  app->Options()->SetStringValue("print_info_string", "yes");
    //  app->Options()->SetStringValue("jac_c_constant", "yes");
    //  app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
    // TODO: Gradient check will mess up your controlPointGrid and so the initialisation...
//    app->Options()->SetStringValue("derivative_test", "first-order");
//    app->Options()->SetStringValue("derivative_test_print_all", "yes");
//    app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);

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
  // print status for the final step
  if (status == Solve_Succeeded) {
    std::cout << std::endl << "*** The problem is solved!" << std::endl;
  } else {
    std::cout << std::endl << "*** The problem FAILED!" << std::endl;
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
