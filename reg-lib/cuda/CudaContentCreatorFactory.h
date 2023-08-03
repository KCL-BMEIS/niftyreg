#pragma once

#include "ContentCreatorFactory.h"
#include "CudaContentCreator.h"
#include "CudaAladinContentCreator.h"
#include "CudaDefContentCreator.h"
#include "CudaF3dContentCreator.h"

class CudaContentCreatorFactory: public ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType& conType) override {
        switch (conType) {
        case ContentType::Aladin:
            return new CudaAladinContentCreator();
        case ContentType::Def:
            return new CudaDefContentCreator();
        case ContentType::F3d:
            return new CudaF3dContentCreator();
        default:
            return new CudaContentCreator();
        }
    }
};
