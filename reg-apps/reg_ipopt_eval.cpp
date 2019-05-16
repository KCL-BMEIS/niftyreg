//
// Created by lf18 on 01/04/19.
//

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

    // Read the reference image
    nifti_image *referenceImage = NULL;
    referenceImage = reg_io_ReadImageFile(CommandLineReaderRegIpoptEval::getInstance().getRefImgPath().c_str());
    if (!referenceImage) {
        throw CouldNotReadInputImage(CommandLineReaderRegIpoptEval::getInstance().getRefImgPath());
    }

    // Compute the logJacobian map
    if (CommandLineReaderRegIpoptEval::getInstance().getLogJacobianFlag()) {
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
        defImage->datatype = NIFTI_TYPE_FLOAT64;
        defImage->data = (void *)calloc(defImage->nvox, defImage->nbyper);
        defImage->intent_code=NIFTI_INTENT_VECTOR;
        memset(defImage->intent_name, 0, 16);
        strcpy(defImage->intent_name,"NREG_TRANS");
        defImage->intent_p1=DEF_FIELD;
        defImage->scl_slope=1.f;
        defImage->scl_inter=0.f;

        // Compute true jacobian fior the Euler method
        nifti_image *jac = reg_spline_GetJacobianFromVelocityGrid(defImage, velocityCPP);
//        fileName = stringFormat("%s/output_logJacobian_euler_level%d.nii.gz",
//                                this->saveDir.c_str(), this->currentLevel+1);
        reg_io_WriteImageFile(jac, CommandLineReaderRegIpoptEval::getInstance().getOutputFilePath().c_str());
        // Free the deformation field
        nifti_image_free(defImage);
    }
    else if (CommandLineReaderRegIpoptEval::getInstance().getLogJacobianFlag()){
        // Create an identity deformation field based on the reference image for the landmark
        nifti_image *defImage = nifti_copy_nim_info(referenceImage);
        defImage->dim[0]=defImage->ndim=5;
        defImage->dim[1]=defImage->nx=1;
        defImage->dim[2]=defImage->ny=1;
        defImage->dim[3]=defImage->nz=1;
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
        defImage->datatype = NIFTI_TYPE_FLOAT64;
        defImage->data = (void *)calloc(defImage->nvox, defImage->nbyper);
        defImage->intent_code=NIFTI_INTENT_VECTOR;
        memset(defImage->intent_name, 0, 16);
        strcpy(defImage->intent_name,"NREG_TRANS");
        defImage->intent_p1=DEF_FIELD;
        defImage->scl_slope=1.f;
        defImage->scl_inter=0.f;


    }
    else {
        std::cout << "options other than computation of the log-Jacobian are not implemented yet." << std::endl;
    }

    nifti_image_free(referenceImage);
    nifti_image_free(velocityCPP);

}


