#ifndef _REG_READWRITEBINARY_H
#define _REG_READWRITEBINARY_H

#include <fstream>      // std::ifstream
#include <stdlib.h>

extern "C++"
void readFloatBinaryArray(char* fileName, int lengthArray, float* outputArray);

#endif
