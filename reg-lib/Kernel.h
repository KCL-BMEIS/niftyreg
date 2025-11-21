#pragma once

#include <iostream>
#include <string>

class Kernel {
public:
    Kernel() = default;
    virtual ~Kernel() = default;

    template <class T>
    T* castTo() { return dynamic_cast<T*>(this); }
};
