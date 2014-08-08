#include "CPUKernelFactory.h"
#include "KernelImpl.h"
#include "CPUKernels.h"
#include "Platform.h"

KernelImpl* CPUKernelFactory::createKernelImpl(std::string name, const Platform& platform, unsigned int dType) const {

	switch( dType ) {
	case NIFTI_TYPE_UINT8:
		registerDataTypes<unsigned char>(name, platform);
		break;
	case NIFTI_TYPE_INT8:
		registerDataTypes<char>(name, platform);
		break;
	case NIFTI_TYPE_UINT16:
		registerDataTypes<unsigned short>(name, platform);
		break;
	case NIFTI_TYPE_INT16:
		registerDataTypes<short>(name, platform);
		break;
	case NIFTI_TYPE_UINT32:
		registerDataTypes<unsigned int>(name, platform);
		break;
	case NIFTI_TYPE_INT32:
		registerDataTypes<int>(name, platform);
		break;
	case NIFTI_TYPE_FLOAT32:
		std::cout << "float32" << std::endl;
		registerDataTypes<float>(name, platform);
		break;
	case NIFTI_TYPE_FLOAT64:
		registerDataTypes<double>(name, platform);
		break;
	default:
		fprintf(stderr, "[NiftyReg ERROR] reg_gaussianSmoothing\tThe image data type is not supported\n");
		exit(1);
	}
	
	
	
	return NULL;


	//put this on the calling function
	/**/
}

template<class T>
KernelImpl* CPUKernelFactory::registerDataTypes(std::string name, const Platform& platform) const {
	if( name == AffineDeformationField3DKernel<void>::Name() ) return new CPUAffineDeformationField3DKernel<T>(name, platform);
	else if( name == CPUConvolutionKernel<void>::Name() ) return new CPUConvolutionKernel<T>(name, platform);
	else return NULL;
}


template KernelImpl* CPUKernelFactory::registerDataTypes<unsigned short>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<unsigned int>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<unsigned char>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<short>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<int>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<char>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<double>(std::string name, const Platform& platform) const;
template KernelImpl* CPUKernelFactory::registerDataTypes<float>(std::string name, const Platform& platform) const;

