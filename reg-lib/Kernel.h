#pragma once

#include <iostream>
#include <string>

class Kernel {
public:
    Kernel() {}
    virtual ~Kernel() {}

    std::string GetName() const;

    template <class T>
    T* castTo() { return dynamic_cast<T*>(this); }
};
