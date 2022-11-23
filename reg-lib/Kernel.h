#pragma once

#include <iostream>
#include <string>

class Kernel {
public:
    Kernel(std::string nameIn) { name = nameIn; }
    virtual ~Kernel() {}

    std::string GetName() const;

    template <class T>
    T* castTo() { return dynamic_cast<T*>(this); }

private:
    std::string name;
};
