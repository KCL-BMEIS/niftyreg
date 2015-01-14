#ifndef KERNEL_H_
#define KERNEL_H_

#include <iostream>
#include <string>

class Kernel {
public:


	Kernel(std::string nameIn){ name = nameIn; }
	virtual ~Kernel(){}

	std::string getName() const;
	std::string name;

	template <class T>
	T* castTo() {
		return dynamic_cast<T*>(this);
	}
};



#endif /*KERNEL_H_*/
