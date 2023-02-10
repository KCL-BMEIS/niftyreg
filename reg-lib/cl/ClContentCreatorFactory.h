#pragma once

#include "ContentCreatorFactory.h"
#include "ClAladinContentCreator.h"

class ClContentCreatorFactory: public ContentCreatorFactory {
public:
    virtual ContentCreator* Produce(const ContentType& conType) override {
        switch (conType) {
        case ContentType::Aladin:
            return new ClAladinContentCreator();
        default:
            reg_print_fct_error("ClContentFactory::Produce");
            reg_print_msg_error("Unsupported content type");
            reg_exit();
        }
    }
};
