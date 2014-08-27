//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// OpenCLInfo.cpp
//
//    This is a simple example that demonstrates use of the clGetInfo* functions, 
//    with particular focus on platforms and their associated devices.

#include <iostream>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <malloc.h> // needed for alloca
#endif // _WIN32
#if defined(linux) || defined(__APPLE__) || defined(__MACOSX)
# include <alloca.h>
#endif // linux
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
// Display information for a particular platform.
// Assumes that all calls to clGetPlatformInfo returns
// a value of type char[], which is valid for OpenCL 1.1.
//
void DisplayPlatformInfo( cl_platform_id id, cl_platform_info name, std::string str ) {
	cl_int errNum;
	std::size_t paramValueSize;

	errNum = clGetPlatformInfo( id, name, 0, NULL, &paramValueSize );
	if (errNum != CL_SUCCESS) {
		std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
		return;
	}

	char * info = (char *) alloca(sizeof(char) * paramValueSize);
	errNum = clGetPlatformInfo( id, name, paramValueSize, info, NULL );
	if (errNum != CL_SUCCESS) {
		std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
		return;
	}

	std::cout << "\t" << str << ":\t" << info << std::endl;
}

template<typename T>
void appendBitfield( T info, T value, std::string name, std::string & str ) {
	if (info & value) {
		if (str.length() > 0) {
			str.append( " | " );
		}
		str.append( name );
	}
}

///
// Display information for a particular device.
// As different calls to clGetDeviceInfo may return
// values of different types a template is used. 
// As some values returned are arrays of values, a templated class is
// used so it can be specialized for this case, see below.
//
template<typename T>
class InfoDevice {
public:
	static void display( cl_device_id id, cl_device_info name, std::string str ) {
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo( id, name, 0, NULL, &paramValueSize );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *) alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo( id, name, paramValueSize, info, NULL );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		// Handle a few special cases
		switch (name) {
		case CL_DEVICE_TYPE: {
			std::string deviceType;

			appendBitfield<cl_device_type>( *(reinterpret_cast<cl_device_type*>( info )), CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", deviceType );

			appendBitfield<cl_device_type>( *(reinterpret_cast<cl_device_type*>( info )), CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", deviceType );

			appendBitfield<cl_device_type>( *(reinterpret_cast<cl_device_type*>( info )), CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", deviceType );

			appendBitfield<cl_device_type>( *(reinterpret_cast<cl_device_type*>( info )), CL_DEVICE_TYPE_DEFAULT, "CL_DEVICE_TYPE_DEFAULT", deviceType );

			std::cout << "\t\t" << str << ":\t" << deviceType << std::endl;
		}
			break;
		case CL_DEVICE_SINGLE_FP_CONFIG: {
			std::string fpType;

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_DENORM, "CL_FP_DENORM", fpType );

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_INF_NAN, "CL_FP_INF_NAN", fpType );

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_ROUND_TO_NEAREST, "CL_FP_ROUND_TO_NEAREST", fpType );

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_ROUND_TO_ZERO, "CL_FP_ROUND_TO_ZERO", fpType );

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_ROUND_TO_INF, "CL_FP_ROUND_TO_INF", fpType );

			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_FMA, "CL_FP_FMA", fpType );

#ifdef CL_FP_SOFT_FLOAT
			appendBitfield<cl_device_fp_config>( *(reinterpret_cast<cl_device_fp_config*>( info )), CL_FP_SOFT_FLOAT, "CL_FP_SOFT_FLOAT", fpType );
#endif

			std::cout << "\t\t" << str << ":\t" << fpType << std::endl;
		}
		case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: {
			std::string memType;

			appendBitfield<cl_device_mem_cache_type>( *(reinterpret_cast<cl_device_mem_cache_type*>( info )), CL_NONE, "CL_NONE", memType );
			appendBitfield<cl_device_mem_cache_type>( *(reinterpret_cast<cl_device_mem_cache_type*>( info )), CL_READ_ONLY_CACHE, "CL_READ_ONLY_CACHE", memType );

			appendBitfield<cl_device_mem_cache_type>( *(reinterpret_cast<cl_device_mem_cache_type*>( info )), CL_READ_WRITE_CACHE, "CL_READ_WRITE_CACHE", memType );

			std::cout << "\t\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_LOCAL_MEM_TYPE: {
			std::string memType;

			appendBitfield<cl_device_local_mem_type>( *(reinterpret_cast<cl_device_local_mem_type*>( info )), CL_GLOBAL, "CL_LOCAL", memType );

			appendBitfield<cl_device_local_mem_type>( *(reinterpret_cast<cl_device_local_mem_type*>( info )), CL_GLOBAL, "CL_GLOBAL", memType );

			std::cout << "\t\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_EXECUTION_CAPABILITIES: {
			std::string memType;

			appendBitfield<cl_device_exec_capabilities>( *(reinterpret_cast<cl_device_exec_capabilities*>( info )), CL_EXEC_KERNEL, "CL_EXEC_KERNEL", memType );

			appendBitfield<cl_device_exec_capabilities>( *(reinterpret_cast<cl_device_exec_capabilities*>( info )), CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL", memType );

			std::cout << "\t\t" << str << ":\t" << memType << std::endl;
		}
			break;
		case CL_DEVICE_QUEUE_PROPERTIES: {
			std::string memType;

			appendBitfield<cl_device_exec_capabilities>( *(reinterpret_cast<cl_device_exec_capabilities*>( info )), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", memType );

			appendBitfield<cl_device_exec_capabilities>( *(reinterpret_cast<cl_device_exec_capabilities*>( info )), CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", memType );

			std::cout << "\t\t" << str << ":\t" << memType << std::endl;
		}
			break;
		default:
			std::cout << "\t\t" << str << ":\t" << *info << std::endl;
			break;
		}
	}
};

///
// Simple trait class used to wrap base types.
//
template<typename T>
class ArrayType {
public:
	static bool isChar() {
		return false;
	}
};

///
// Specialized for the char (i.e. null terminated string case).
//
template<>
class ArrayType<char> {
public:
	static bool isChar() {
		return true;
	}
};

///
// Specialized instance of class InfoDevice for array types.
//
template<typename T>
class InfoDevice<ArrayType<T> > {
public:
	static void display( cl_device_id id, cl_device_info name, std::string str ) {
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo( id, name, 0, NULL, &paramValueSize );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *) alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo( id, name, paramValueSize, info, NULL );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		if (ArrayType<T>::isChar()) {
			std::cout << "\t" << str << ":\t" << info << std::endl;
		} else if (name == CL_DEVICE_MAX_WORK_ITEM_SIZES) {
			cl_uint maxWorkItemDimensions;

			errNum = clGetDeviceInfo( id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL );
			if (errNum != CL_SUCCESS) {
				std::cerr << "Failed to find OpenCL device info " << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS." << std::endl;
				return;
			}

			std::cout << "\t" << str << ":\t";
			for (cl_uint i = 0; i < maxWorkItemDimensions; i++) {
				std::cout << info[i] << " ";
			}
			std::cout << std::endl;
		}
	}
};

///
//  Enumerate platforms and display information about them 
//  and their associated devices.
//
void displayInfo( void ) {
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id * platformIds;
	cl_context context = NULL;

	// First, query the total number of platforms
	errNum = clGetPlatformIDs( 0, NULL, &numPlatforms );
	if (errNum != CL_SUCCESS || numPlatforms <= 0) {
		std::cerr << "Failed to find any OpenCL platform." << std::endl;
		return;
	}

	// Next, allocate memory for the installed plaforms, and qeury 
	// to get the list.
	platformIds = (cl_platform_id *) alloca(sizeof(cl_platform_id) * numPlatforms);
	// First, query the total number of platforms
	errNum = clGetPlatformIDs( numPlatforms, platformIds, NULL );
	if (errNum != CL_SUCCESS) {
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return;
	}

	std::cout << "Number of platforms: \t" << numPlatforms << std::endl;
	// Iterate through the list of platforms displaying associated information
	for (cl_uint i = 0; i < numPlatforms; i++) {
		std::cout<<"=================================================================================================="<<std::endl;
		// First we display information associated with the platform
		DisplayPlatformInfo( platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE" );
		DisplayPlatformInfo( platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION" );
		DisplayPlatformInfo( platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR" );
		DisplayPlatformInfo( platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS" );

		// Now query the set of devices associated with the platform
		cl_uint numDevices;
		errNum = clGetDeviceIDs( platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL devices." << std::endl;
			return;
		}

		cl_device_id * devices = (cl_device_id *) alloca(sizeof(cl_device_id) * numDevices);
		errNum = clGetDeviceIDs( platformIds[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL );
		if (errNum != CL_SUCCESS) {
			std::cerr << "Failed to find OpenCL devices." << std::endl;
			return;
		}

		std::cout << "\tNumber of devices: \t" << numDevices << std::endl;
		// Iterate through each device, displaying associated information
		for (cl_uint j = 0; j < numDevices; j++) {
			std::cout<<"------------------------------------------------------------------------------------  "<<std::endl;

			InfoDevice<cl_device_type>::display( devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" );

			InfoDevice<ArrayType<size_t> >::display( devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE" );

#ifdef CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF" );
#endif

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS" );

			InfoDevice<cl_ulong>::display( devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE" );

			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_READ_IMAGE_ARGS, "CL_DEVICE_MAX_READ_IMAGE_ARGS" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "CL_DEVICE_MAX_WRITE_IMAGE_ARGS" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, "CL_DEVICE_IMAGE2D_MAX_WIDTH" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH, "CL_DEVICE_IMAGE2D_MAX_WIDTH" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT, "CL_DEVICE_IMAGE2D_MAX_HEIGHT" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE3D_MAX_WIDTH, "CL_DEVICE_IMAGE3D_MAX_WIDTH" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE3D_MAX_HEIGHT, "CL_DEVICE_IMAGE3D_MAX_HEIGHT" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_IMAGE3D_MAX_DEPTH, "CL_DEVICE_IMAGE3D_MAX_DEPTH" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_SAMPLERS, "CL_DEVICE_MAX_SAMPLERS" );

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_MAX_PARAMETER_SIZE, "CL_DEVICE_MAX_PARAMETER_SIZE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE" );

			InfoDevice<cl_device_fp_config>::display( devices[j], CL_DEVICE_SINGLE_FP_CONFIG, "CL_DEVICE_SINGLE_FP_CONFIG" );

			InfoDevice<cl_device_mem_cache_type>::display( devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE" );

			InfoDevice<cl_ulong>::display( devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE" );

			InfoDevice<cl_ulong>::display( devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE" );

			InfoDevice<cl_ulong>::display( devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" );

			InfoDevice<cl_uint>::display( devices[j], CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS" );

			InfoDevice<cl_device_local_mem_type>::display( devices[j], CL_DEVICE_LOCAL_MEM_TYPE, "CL_DEVICE_LOCAL_MEM_TYPE" );

			InfoDevice<cl_ulong>::display( devices[j], CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE" );

			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_ERROR_CORRECTION_SUPPORT, "CL_DEVICE_ERROR_CORRECTION_SUPPORT" );

#ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, "CL_DEVICE_HOST_UNIFIED_MEMORY" );
#endif

			InfoDevice<std::size_t>::display( devices[j], CL_DEVICE_PROFILING_TIMER_RESOLUTION, "CL_DEVICE_PROFILING_TIMER_RESOLUTION" );

			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE" );

			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE" );

			InfoDevice<cl_bool>::display( devices[j], CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE" );

			InfoDevice<cl_device_exec_capabilities>::display( devices[j], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES" );

			InfoDevice<cl_command_queue_properties>::display( devices[j], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES" );

			InfoDevice<cl_platform_id>::display( devices[j], CL_DEVICE_PLATFORM, "CL_DEVICE_PLATFORM" );

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_NAME, "CL_DEVICE_NAME" );

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR" );

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DRIVER_VERSION, "CL_DRIVER_VERSION" );

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_PROFILE, "CL_DEVICE_PROFILE" );

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_VERSION, "CL_DEVICE_VERSION" );

#ifdef CL_DEVICE_OPENCL_C_VERSION
			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_OPENCL_C_VERSION, "CL_DEVICE_OPENCL_C_VERSION" );
#endif

			InfoDevice<ArrayType<char> >::display( devices[j], CL_DEVICE_EXTENSIONS, "CL_DEVICE_EXTENSIONS" );

			std::cout << std::endl << std::endl;
		}
	}
}

///
//	main() for OpenCLInfo example
//
int main( int argc, char** argv ) {
	cl_context context = 0;

	displayInfo();

	return 0;
}
