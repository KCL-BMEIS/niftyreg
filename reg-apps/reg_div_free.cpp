//
// Created by lf18 on 30/03/19.
//

#include "command_line_reader_div_free_projection.h"
#include "_div_free_projection.h"
#include "_reg_ReadWriteImage.h"
#include "nifti1.h"
#include "exception.h"
#include "IpIpoptApplication.hpp"
#include <float.h>
#include <cstdio>
#include <sys/stat.h>

int main(int argc, char** argv) {
    // Set time variables for logs
    time_t start;
    time_t end;
    time(&start);

    // Read the command line options
    CommandLineReaderDivFreeProjection::getInstance().processCmdLineOptions(argc, argv);

    // If the user asks for help print help and close program
    if (CommandLineReaderDivFreeProjection::getInstance().justHelp()) {
        CommandLineReaderDivFreeProjection::getInstance().printUsage(std::cout);
        return EXIT_SUCCESS;
    }

    // Read the velocity vector field to make divergence-free
    nifti_image *velocityCPP = NULL;
    velocityCPP = reg_io_ReadImageFile(CommandLineReaderDivFreeProjection::getInstance().getVelocityFilePath().c_str());
    if (!velocityCPP) {
        throw CouldNotReadInputImage(CommandLineReaderDivFreeProjection::getInstance().getVelocityFilePath());
    }

    // create the div_free_projection object
    SmartPtr<div_free_projection<double>> OPTIM = new div_free_projection<double>(velocityCPP);

    // Set the path to save the output divergence-free velocity vector field
    OPTIM->set_save_path(CommandLineReaderDivFreeProjection::getInstance().getOutputFilePath());

    // Set parameters for the optimisation
    int max_iter = 200;
    ApplicationReturnStatus status;

    // create the Ipopt application that will run the optimisation for the euclidian projection
    SmartPtr<IpoptApplication> app = new IpoptApplication();

    // tell the application that the constraints are linear
    app->Options()->SetStringValue("jac_c_constant", "yes");
    app->Options()->SetStringValue("jac_d_constant", "yes");

    // the Hessian also constant (Identity for an euclidian projection)
    app->Options()->SetStringValue("hessian_constant", "yes");

    // choose the linear solver
    app->Options()->SetStringValue("linear_solver", "ma57");
    app->Options()->SetStringValue("ma57_automatic_scaling", "no");

    // set the print options
    app->Options()->SetStringValue("print_info_string", "yes");
    app->Options()->SetStringValue("print_timing_statistics", "yes");

    // set the convergence criteria options
    app->Options()->SetNumericValue("tol", 1e-3);  // default scale*1e-4
    app->Options()->SetNumericValue("acceptable_obj_change_tol", 1e-6);  // stop criteria based on objective
    app->Options()->SetNumericValue("acceptable_tol", 1.);  // default scale*1e-3
//    app->Options()->SetNumericValue("acceptable_compl_inf_tol", 10000.);  // default 0.01
    app->Options()->SetIntegerValue("acceptable_iter", 15);  // default 15
    app->Options()->SetIntegerValue("max_iter", max_iter);

    // Intialize the IpoptApplication and process the options
    app->Initialize();

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(OPTIM);

    time(&end);
    int minutes=(int)floorf((end - start) / 60.0f);
    int seconds=(int)(end - start - 60*minutes);
    std::cout << "Euclidian projection performed in " << minutes << " min " << seconds << " sec" <<std::endl;
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
    std::cout << "Have a good day!" << std::endl;

    return 0;
}
