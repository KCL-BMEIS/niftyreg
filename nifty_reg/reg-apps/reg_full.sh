#
#  reg_full.sh
#
#  Created by Marc Modat on 14/09/2009.
#  Copyright (c) 2009, University College London. All rights reserved.
#  Centre for Medical Image Computing (CMIC)
#  See the LICENSE.txt file in the nifty_reg root folder
#
#
#!/bin/sh

#===================================================================
f_output_usage()
#===================================================================
{
	echo -e "\n*************************************************************************"
	echo -e "This simple script performs first an affine registration using reg_aladin"
	echo -e "and then a non-rigid registration using reg_f3d"
	echo -e "Usage:\n\t$0 -target <fileName> -source <fileName> [options]"
	echo -e "[options]:"
	echo -e "-target <string>\tFilename of the target image (Analyse or nifti format)"
	echo -e "-source <string>\tFilename of the source image (Analyse or nifti format)"
	echo -e "-aff <string>\t\tFilename of the result affine matrix(text file)"
	echo -e "-affRes <string>\tFilename of the result affine image"
	echo -e "-nrrCpp <string>\tFilename of the non-rigid control Point grid"
	echo -e "-nrrRes <string>\tFilename of the non-rigid result image"
	echo -e "*************************************************************************\n"
}


#===================================================================
# MAIN
#===================================================================
echo ""
#arguments
TARGET="\0";
SOURCE="\0";
AFFINE_RES="affine_result_img.nii";
AFFINE_MAT="affine_result_mat.nii";
NRR_RES="nrr_result_img.nii";
NRR_CPP="nrr_result_cpp.nii";

#parsing argument
while [ $# != 0 ]; do
	flag="$1"
	case "$flag" in
		-target) if [ $# -gt 1 ];
		then
			TARGET="$2"
			shift
		fi
		;;
		-source) if [ $# -gt 1 ];
		then
			SOURCE="$2"
			shift
		fi
		;;
		-aff) if [ $# -gt 1 ];
		then
			AFFINE_MAT="$2"
			shift
		fi
		;;
		-affRes) if [ $# -gt 1 ];
		then
			AFFINE_RES="$2"
			shift
		fi
		;;
		-nrrCpp) if [ $# -gt 1 ];
		then
			NRR_CPP="$2"
			shift
		fi
		;;
		-nrrRes) if [ $# -gt 1 ];
		then
			NRR_RES="$2"
			shift
		fi
		;;
		*) echo "Unrecognized flag or argument: $flag"
		f_output_usage
		exit
		;;
	esac
	shift
done

# print out of the parameters
echo -e "Target image\t\t${TARGET}"
echo -e "Source image\t\t${SOURCE}"
echo -e "Affine result\t\t${AFFINE_RES}"
echo -e "Affine matrix\t\t${AFFINE_MAT}"
echo -e "Non-rigid result\t${NRR_RES}"
echo -e "Non-rigid cpp\t\t${NRR_CPP}"

if [ $TARGET == "\0" ]
then
	echo -e "The target image has to be defined"
	f_output_usage
	exit
fi
if [ $SOURCE == "\0" ]
then
	echo -e "The source image has to be defined"
	f_output_usage
	exit
fi

#Affine registration
reg_aladin \
	-target $TARGET \
	-source $SOURCE \
	-aff $AFFINE_MAT \
	-result $AFFINE_RES
#Non-Rigid registration
reg_f3d \
	-target $TARGET \
	-source $SOURCE \
	-aff $AFFINE_MAT \
	-cpp $NRR_CPP \
	-result $NRR_RES

exit