#include "_reg_ReadWriteBinary.h"

void readFloatBinaryArray(char* fileName, int lengthArray, float* outputArray) {
    FILE* infile;
    infile=fopen(fileName,"rb");
    float currentValue;
    for (int i =0;i<lengthArray;i++) {
        fread((void*)(&currentValue), sizeof(currentValue), 1, infile);
        outputArray[i]=currentValue;
    }
}
