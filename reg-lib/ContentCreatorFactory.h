#pragma once

#include "ContentCreator.h"
#include "AladinContentCreator.h"
#include "DefContentCreator.h"
#include "F3dContentCreator.h"

enum class ContentType { Base, Aladin, Def, F3d };

class ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType& conType) {
        switch (conType) {
        case ContentType::Aladin:
            return new AladinContentCreator();
        case ContentType::Def:
            return new DefContentCreator();
        case ContentType::F3d:
            return new F3dContentCreator();
        default:
            return new ContentCreator();
        }
    }
};
