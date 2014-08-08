#include "KernelImpl.h"

using namespace std;

KernelImpl::KernelImpl(string name, const Platform& platform) : name(name), platform(&platform), referenceCount(1) {
}

std::string KernelImpl::getName() const {
    return name;
}

const Platform& KernelImpl::getPlatform() {
    return *platform;
}

