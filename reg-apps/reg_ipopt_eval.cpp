//
// Created by lf18 on 01/04/19.
//

#include "_reg_localTrans_jac.h"
#include "_reg_tools.h"
#include "command_line_reader_reg_ipopt_eval.h"
#include "_reg_f3d2_ipopt.h"
#include <float.h>
#include <cstdio>
#include <sys/stat.h>
#include "exception.h"

int main(int argc, char** argv) {
    // Read the command line options
    CommandLineReaderRegIpoptEval::getInstance().processCmdLineOptions(argc, argv);

    // If the user asks for help print help and close program
    if (CommandLineReaderRegIpoptEval::getInstance().justHelp()) {
        CommandLineReaderRegIpoptEval::getInstance().printUsage(std::cout);
        return EXIT_SUCCESS;
    }

    // Read the velocity vector field to make divergence-free
    nifti_image *velocityCPP = NULL;
    velocityCPP = reg_io_ReadImageFile(CommandLineReaderRegIpoptEval::getInstance().getVelocityFilePath().c_str());
    if (!velocityCPP) {
        throw CouldNotReadInputImage(CommandLineReaderRegIpoptEval::getInstance().getVelocityFilePath());
    }

    //test
//    reg_getDisplacementFromDeformation(velocityCPP);
//    reg_tools_multiplyValueToImage(velocityCPP, velocityCPP, 0.f);
//    reg_tools_addValueToImage(velocityCPP, velocityCPP, 1.0f);
//    reg_getDeformationFromDisplacement(velocityCPP);

    // change datatype of the velocity field to double precision if necessary
    if (velocityCPP->datatype == NIFTI_TYPE_FLOAT32) {
        reg_tools_changeDatatype<double>(velocityCPP);
    }

    // Read the reference image
    nifti_image *referenceImage = NULL;
    referenceImage = reg_io_ReadImageFile(CommandLineReaderRegIpoptEval::getInstance().getRefImgPath().c_str());
    if (!referenceImage) {
        throw CouldNotReadInputImage(CommandLineReaderRegIpoptEval::getInstance().getRefImgPath());
    }

    // Compute the logJacobian map
    if (CommandLineReaderRegIpoptEval::getInstance().getLogJacobianFlag()) {
        // only cubic B-splines and divergence conforming B-splines are supported
        assert (splineControlPoint->intent_p1 == SPLINE_VEL_GRID ||
                splineControlPoint->intent_p1 == DIV_CONFORMING_VEL_GRID);
        // Create an identity deformation field based on the reference image
        nifti_image *defImage = nifti_copy_nim_info(referenceImage);
        defImage->dim[0]=defImage->ndim=5;
        defImage->dim[1]=defImage->nx=referenceImage->nx;
        defImage->dim[2]=defImage->ny=referenceImage->ny;
        defImage->dim[3]=defImage->nz=referenceImage->nz;
        defImage->dim[4]=defImage->nt=1;
        defImage->pixdim[4]=defImage->dt=1.0;
        if(referenceImage->nz==1)
            defImage->dim[5]=defImage->nu=2;
        else defImage->dim[5]=defImage->nu=3;
        defImage->pixdim[5]=defImage->du=1.0;
        defImage->dim[6]=defImage->nv=1;
        defImage->pixdim[6]=defImage->dv=1.0;
        defImage->dim[7]=defImage->nw=1;
        defImage->pixdim[7]=defImage->dw=1.0;
        defImage->nvox =
                (size_t)defImage->nx *
                (size_t)defImage->ny *
                (size_t)defImage->nz *
                (size_t)defImage->nt *
                (size_t)defImage->nu;
        defImage->nbyper = sizeof(double);
        defImage->datatype = velocityCPP->datatype;
        defImage->data = (void *)calloc(defImage->nvox, defImage->nbyper);
        defImage->intent_code=NIFTI_INTENT_VECTOR;
        memset(defImage->intent_name, 0, 16);
        strcpy(defImage->intent_name,"NREG_TRANS");
        defImage->intent_p1=DEF_FIELD;
        defImage->scl_slope=1.f;
        defImage->scl_inter=0.f;

        // Compute true jacobian for the Euler method
        std::cout << "Compute the log Jacobian map of the velocity field using an Euler integration"
            << std::endl;
        nifti_image *jac = reg_spline_GetLogJacobianFromVelocityGrid(defImage, velocityCPP);
        // save the log Jacobian map
        reg_io_WriteImageFile(jac, CommandLineReaderRegIpoptEval::getInstance().getOutputFilePath().c_str());
        std::cout << "The log Jacobian map " << CommandLineReaderRegIpoptEval::getInstance().getOutputFilePath()
            << " has been saved." << std::endl;
        // Free the deformation field
        nifti_image_free(defImage);
    }
    else {
        std::cout << "options other than computation of the log-Jacobian are not implemented yet." << std::endl;
    }

    nifti_image_free(referenceImage);
    nifti_image_free(velocityCPP);

}
