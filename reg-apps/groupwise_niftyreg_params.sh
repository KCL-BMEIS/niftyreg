#!/bin/sh

############################################################################
###################### PARAMETERS THAT CAN BE CHANGED ######################
############################################################################
# Array that contains the input images to create the atlas
export IMG_INPUT=(`ls /path/to/all/your/images_*.nii`)
export IMG_INPUT_MASK= # leave empty to not use floating masks

# template image to use to initialise the atlas creation
export TEMPLATE=`ls ${IMG_INPUT[0]}`
export TEMPLATE_MASK= # leave empty to not use a reference mask

# folder where the result images will be saved
export RES_FOLDER=`pwd`/groupwise_result

# argument to use for the affine (reg_aladin)
export AFFINE_args="-omp 4"
# argument to use for the non-rigid registration (reg_f3d)
export NRR_args="-omp 4"

# number of affine loop to perform - Note that the first step is always rigid
export AFF_IT_NUM=5
# number of non-rigid loop to perform
export NRR_IT_NUM=10

# grid engine arguments
export QSUB_CMD="qsub -l h_rt=05:00:00 -l tmem=0.9G -l h_vmem=0.9G -l vf=0.9G -l s_stack=10240  -j y -S /bin/csh -b y -cwd -V -R y -pe smp 4"
############################################################################
