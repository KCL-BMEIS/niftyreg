#pragma once

#include "ContentCreatorFactory.h"
#include "ClAladinContentCreator.h"

class ClContentCreatorFactory: public ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType conType) override {
        switch (conType) {
        case ContentType::Aladin:
            return new ClAladinContentCreator();
        default:
            NR_FATAL_ERROR("Unsupported content type");
            return nullptr;
        }
    }
};
