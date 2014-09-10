#ifndef CONTEXT_H_
#define CONTEXT_H_



#include <ctime>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>
#include "Kernel.h"
#include "Platform.h"



class ContextImpl;
class Platform;



class  Context {
public:

	Context();
	Context(Platform *platformIn);
	//Context(const System& system, Integrator& integrator);

	//Context(const System& system, Integrator& integrator, Platform& platform);

	//Context(const System& system, Integrator& integrator, Platform& platform, const std::map<std::string, std::string>& properties);
	~Context();

		//get system
	//get optimizer
 
	/**
	 * Get the Platform being used for calculations.
	 */
	const Platform& getPlatform() const;

	//Platform& getPlatform();

	/**
	 * Set the current time of the simulation (in picoseconds).
	 */
	void setTime(double time);
	double getParameter(const std::string& name) const;
	void setParameter(const std::string& name, double value);

	void shout();
	void createKernels(const unsigned int dType);

	Kernel affineTransformation3DKernel, convolutionKernel, blockMatchingKernel, optimiseKernel, resamplingKernel;
	Platform* platform;

};


#endif /*CONTEXT_H_*/
