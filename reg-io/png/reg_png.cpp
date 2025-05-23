/*
 *  reg_png.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012-2018, University College London
 *  Copyright (c) 2018, NiftyReg Developers.
 *  All rights reserved.
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "reg_png.h"
#include "png.h"

using uch = unsigned char;
using ulg = unsigned long;

/* *************************************************************** */
nifti_image *reg_io_readPNGfile(const char *pngFileName, bool readData) {
    // We first read the png file
    FILE *pngFile = nullptr;
    pngFile = fopen(pngFileName, "rb");
    if (pngFile == nullptr)
        NR_FATAL_ERROR("Can not open the png file: "s + pngFileName);

    uch sig[8];
    if (!fread(sig, 1, 8, pngFile))
        NR_FATAL_ERROR("Error when reading the png file: "s + pngFileName);
    if (!png_check_sig(sig, 8))
        NR_FATAL_ERROR("The png file is corrupted: "s + pngFileName);
    rewind(pngFile);

    png_structp pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!pngPtr)
        NR_FATAL_ERROR("Error when reading the png file - out of memory");

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (!infoPtr) {
        png_destroy_read_struct(&pngPtr, nullptr, nullptr);
        NR_FATAL_ERROR("Error when reading the png file - out of memory");
    }

    png_init_io(pngPtr, pngFile);
    png_read_info(pngPtr, infoPtr);

    png_uint_32 width, height;
    int bitDepth, colorType;
    png_get_IHDR(pngPtr, infoPtr, &width, &height, &bitDepth, &colorType, nullptr, nullptr, nullptr);

    int channels;
    ulg rowBytes;

    if (colorType == PNG_COLOR_TYPE_PALETTE)
        png_set_expand(pngPtr);
    if (colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8)
        png_set_expand(pngPtr);
    if (png_get_valid(pngPtr, infoPtr, PNG_INFO_tRNS))
        png_set_expand(pngPtr);

    if (bitDepth == 16)
        png_set_strip_16(pngPtr);
    if (colorType == PNG_COLOR_TYPE_GRAY ||
        colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(pngPtr);

    unique_ptr<png_bytep[]> rowPointers(new png_bytep[height]);

    png_read_update_info(pngPtr, infoPtr);

    rowBytes = png_get_rowbytes(pngPtr, infoPtr);
    channels = (int)png_get_channels(pngPtr, infoPtr);

    if (channels > 3)
        NR_WARN_WFCT("The PNG file has " << channels << " channels. Only the first three are considered for RGB to gray conversion.");
    else if (channels == 2)
        NR_WARN_WFCT("The PNG file has 2 channels. They will be average into one single channel");

    const int dim[8] = { 2, static_cast<int>(width), static_cast<int>(height), 1, 1, 1, 1, 1 };
    nifti_image *niiImage = nullptr;
    if (readData) {

        uch *image_data = static_cast<uch*>(malloc(width * height * channels * sizeof(uch)));
        if (image_data == nullptr)
            NR_FATAL_ERROR("Error while allocating memory for the png file: "s + pngFileName);

        for (png_uint_32 i = 0; i < height; i++)
            rowPointers[i] = image_data + i * rowBytes;

        png_read_image(pngPtr, rowPointers.get());
        png_read_end(pngPtr, nullptr);

        niiImage = nifti_make_new_nim(dim, NIFTI_TYPE_UINT8, true);
        uch *niiPtr = static_cast<uch*>(niiImage->data);
        for (size_t i = 0; i < niiImage->nvox; ++i) niiPtr[i] = 0;
        // Define some weight to create a gray scale image
        float rgb2grayWeight[3];
        if (channels == 1) {
            rgb2grayWeight[0] = 1;
        } else if (channels == 2) {
            rgb2grayWeight[0] = 0.5;
            rgb2grayWeight[1] = 0.5;
        }
        if (channels >= 3) {  // rgb to y
            rgb2grayWeight[0] = 0.299;
            rgb2grayWeight[1] = 0.587;
            rgb2grayWeight[2] = 0.114;
        }
        for (int c = 0; c < (channels < 3 ? channels : 3); c++)
            for (png_uint_32 h = 0; h < height; h++)
                for (png_uint_32 w = 0; w < width; w++)
                    niiPtr[h * niiImage->nx + w] += static_cast<uch>((float)rowPointers[h][w * channels + c] * rgb2grayWeight[c]);
    } else {
        niiImage = nifti_make_new_nim(dim, NIFTI_TYPE_UINT8, false);
    }
    png_destroy_read_struct(&pngPtr, &infoPtr, nullptr);
    fclose(pngFile);

    nifti_set_filenames(niiImage, pngFileName, 0, 0);
    return niiImage;
}
/* *************************************************************** */
void reg_io_writePNGfile(nifti_image *image, const char *filename) {
    // We first check the nifti image dimension
    if (image->nz > 1 || image->nt > 1 || image->nu > 1 || image->nv > 1 || image->nw > 1)
        NR_FATAL_ERROR("Image with dimension larger than 2 can be saved as png");

    // Check the min and max values of the nifti image
    const auto [minValue, maxValue] = NiftiImage(image).data().minmax();

    // Rescale the image intensities if they are outside of the range
    if (minValue < 0 || maxValue > 255) {
        reg_intensityRescale(image, 0, 0, 255);
        NR_WARN_WFCT("The image intensities have been rescaled from [" << minValue << " " << maxValue << "] to [0 255].");
    }

    // The nifti image is converted as unsigned char if required
    if (image->datatype != NIFTI_TYPE_UINT8)
        reg_tools_changeDatatype<uch>(image);

    // Create pointer the nifti image data
    uch *niiImgPtr = static_cast<uch*>(image->data);

    // Check first if the png file can be writen
    FILE *fp = fopen(filename, "wb");
    if (!fp)
        NR_FATAL_ERROR("The png file can not be written: "s + filename);

    // The png file structures are created
    png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (pngPtr == nullptr)
        NR_FATAL_ERROR("The png pointer could not be created");

    png_infop infoPtr = png_create_info_struct(pngPtr);
    if (infoPtr == nullptr)
        NR_FATAL_ERROR("The png structure could not be created");

    // Set the png header information
    png_set_IHDR(pngPtr,
                 infoPtr,
                 image->nx, // width
                 image->ny, // height
                 8, // depth
                 PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    // The rows of the png are intialised
    png_byte **rowPointers = static_cast<png_byte**>(png_malloc(pngPtr, image->ny * sizeof(png_byte*)));
    // The data are copied over from the nifti structure to the png structure
    size_t niiIndex = 0;
    for (int y = 0; y < image->ny; y++) {
        png_byte *row = static_cast<png_byte*>(png_malloc(pngPtr, sizeof(uch) * image->nx));
        rowPointers[y] = row;
        for (int x = 0; x < image->nx; x++)
            *row++ = niiImgPtr[niiIndex++];
    }
    // Write the image data to the file
    png_init_io(pngPtr, fp);
    png_set_rows(pngPtr, infoPtr, rowPointers);
    png_write_png(pngPtr, infoPtr, PNG_TRANSFORM_IDENTITY, nullptr);
    // Free the allocated png arrays
    for (int y = 0; y < image->ny; y++)
        png_free(pngPtr, rowPointers[y]);
    png_free(pngPtr, rowPointers);
    png_destroy_write_struct(&pngPtr, &infoPtr);
    // Finally close the file on the hard-drive
    fclose(fp);
}
/* *************************************************************** */
