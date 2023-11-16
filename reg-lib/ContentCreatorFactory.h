#pragma once

#include "ContentCreator.h"
#include "AladinContentCreator.h"
#include "DefContentCreator.h"
#include "F3dContentCreator.h"
#include "F3d2ContentCreator.h"

enum class ContentType { Base, Aladin, Def, F3d, F3d2 };

class ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType conType) {
        switch (conType) {
        case ContentType::Base:
            return new ContentCreator();
        case ContentType::Aladin:
            return new AladinContentCreator();
        case ContentType::Def:
            return new DefContentCreator();
        case ContentType::F3d:
            return new F3dContentCreator();
        case ContentType::F3d2:
            return new F3d2ContentCreator();
        default:
            NR_FATAL_ERROR("Unsupported content type");
            return nullptr;
        }
    }
};
