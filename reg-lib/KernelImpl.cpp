#include "KernelImpl.h"

using namespace std;

KernelImpl::KernelImpl(string nameIn, const Platform& platform) : name(nameIn), platform(&platform), referenceCount(1) {
}

std::string KernelImpl::getName() const {
    return name;
}

const Platform& KernelImpl::getPlatform() {
    return *platform;
}

