#pragma once

#include "ContentCreatorFactory.h"
#include "CudaContentCreator.h"
#include "CudaAladinContentCreator.h"
#include "CudaDefContentCreator.h"
#include "CudaF3dContentCreator.h"
#include "CudaF3d2ContentCreator.h"

class CudaContentCreatorFactory: public ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType conType) override {
        switch (conType) {
        case ContentType::Base:
            return new CudaContentCreator();
        case ContentType::Aladin:
            return new CudaAladinContentCreator();
        case ContentType::Def:
            return new CudaDefContentCreator();
        case ContentType::F3d:
            return new CudaF3dContentCreator();
        case ContentType::F3d2:
            return new CudaF3d2ContentCreator();
        default:
            NR_FATAL_ERROR("Unsupported content type");
            return nullptr;
        }
    }
};
