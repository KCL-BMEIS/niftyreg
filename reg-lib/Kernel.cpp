

#include "Kernel.h"

using namespace std;

Kernel::Kernel() : impl(0) {
}

Kernel::Kernel(KernelImpl* impl) : impl(impl) {
}

Kernel::Kernel(const Kernel& copy) : impl(copy.impl) {
    if (impl)
        impl->referenceCount++;
}

Kernel::~Kernel() {
    if (impl) {
        impl->referenceCount--;
        if (impl->referenceCount == 0)
            delete impl;
    }
}

Kernel& Kernel::operator=(const Kernel& copy) {
    if (impl) {
        impl->referenceCount--;
        if (impl->referenceCount == 0)
            delete impl;
    }
    impl = copy.impl;
    if (impl)
        impl->referenceCount++;
    return *this;
}

string Kernel::getName() const {
    return impl->getName();
}

const KernelImpl& Kernel::getImpl() const {
    return *impl;
}

KernelImpl& Kernel::getImpl() {
    return *impl;
}
