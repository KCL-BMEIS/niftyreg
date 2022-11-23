#pragma once

#include <fstream>      // std::ifstream
#include <stdlib.h>

extern "C++"
void readFloatBinaryArray(const char* fileName, int lengthArray, float* outputArray);
void readIntBinaryArray(const char* fileName, int lengthArray, int* outputArray);
