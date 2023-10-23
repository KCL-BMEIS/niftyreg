/*
 *  _reg_ReadWriteImage.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include <filesystem>

/* *************************************************************** */
void reg_hack_filename(nifti_image *image, std::string filename) {
    filename.append("\0");
    // Free the char arrays if already allocated
    if (image->fname) free(image->fname);
    if (image->iname) free(image->iname);
    // Allocate the char arrays
    image->fname = (char *)malloc((filename.size() + 1) * sizeof(char));
    image->iname = (char *)malloc((filename.size() + 1) * sizeof(char));
    // Copy the new name in the char arrays
    strcpy(image->fname, filename.c_str());
    strcpy(image->iname, filename.c_str());
}
/* *************************************************************** */
int reg_io_checkFileFormat(const std::string& filename) {
    // Nifti format is used by default
    // Check the extention of the provided filename
    if (filename.find(".nii.gz") != std::string::npos)
        return NR_NII_FORMAT;
    else if (filename.find(".nii") != std::string::npos)
        return NR_NII_FORMAT;
    else if (filename.find(".hdr") != std::string::npos)
        return NR_NII_FORMAT;
    else if (filename.find(".img.gz") != std::string::npos)
        return NR_NII_FORMAT;
    else if (filename.find(".img") != std::string::npos)
        return NR_NII_FORMAT;
    else if (filename.find(".png") != std::string::npos)
        return NR_PNG_FORMAT;
#ifdef USE_NRRD
    else if (filename.find(".nrrd") != std::string::npos)
        return NR_NRRD_FORMAT;
    else if (filename.find(".nhdr") != std::string::npos)
        return NR_NRRD_FORMAT;
#endif
    else {
        NR_WARN_WFCT("No filename extension provided - the Nifti library is used by default");
    }

    return NR_NII_FORMAT;
}
/* *************************************************************** */
nifti_image* reg_io_ReadImageFile(const char *filename) {
    // First read the file format in order to use the correct library
    const int fileFormat = reg_io_checkFileFormat(filename);

    // Create the nifti image pointer
    nifti_image *image = nullptr;

    // Read the image and convert it to nifti format if required
    switch (fileFormat) {
    case NR_NII_FORMAT:
        image = nifti_image_read(filename, true);
        reg_hack_filename(image, filename);
        break;
    case NR_PNG_FORMAT:
        image = reg_io_readPNGfile(filename, true);
        reg_hack_filename(image, filename);
        break;
#ifdef USE_NRRD
    case NR_NRRD_FORMAT:
        Nrrd *nrrdImage = reg_io_readNRRDfile(filename);
        image = reg_io_nrdd2nifti(nrrdImage);
        nrrdNuke(nrrdImage);
        reg_hack_filename(image, filename);
        break;
#endif
    }
    reg_checkAndCorrectDimension(image);

    // Return the nifti image
    return image;
}
/* *************************************************************** */
nifti_image* reg_io_ReadImageHeader(const char *filename) {
    // First read the file format in order to use the correct library
    const int fileFormat = reg_io_checkFileFormat(filename);

    // Create the nifti image pointer
    nifti_image *image = nullptr;

    // Read the image and convert it to nifti format if required
    switch (fileFormat) {
    case NR_NII_FORMAT:
        image = nifti_image_read(filename, false);
        break;
    case NR_PNG_FORMAT:
        image = reg_io_readPNGfile(filename, false);
        reg_hack_filename(image, filename);
        break;
#ifdef USE_NRRD
    case NR_NRRD_FORMAT:
        Nrrd *nrrdImage = reg_io_readNRRDfile(filename);
        image = reg_io_nrdd2nifti(nrrdImage);
        nrrdNuke(nrrdImage);
        reg_hack_filename(image, filename);
        break;
#endif
    }
    reg_checkAndCorrectDimension(image);

    // Return the nifti image
    return image;
}
/* *************************************************************** */
void reg_io_WriteImageFile(nifti_image *image, const char *filename) {
    // Check if the specified directory exists
    std::filesystem::path p(filename);
    p = p.parent_path();
    if (!std::filesystem::exists(p) && p != std::filesystem::path())
        NR_FATAL_ERROR("The specified folder to save the following file does not exist: "s + filename);

    // First read the file format in order to use the correct library
    int fileFormat = reg_io_checkFileFormat(filename);

    // Check if the images can be saved as a png file
    std::string fname;
    if ((image->nz > 1 ||
         image->nt > 1 ||
         image->nu > 1 ||
         image->nv > 1 ||
         image->nw > 1) &&
        fileFormat == NR_PNG_FORMAT) {
        // If the image has more than two dimension,
        // the filename is converted to nifti
        fname = filename;
        fname.replace(fname.find(".png"), 4, ".nii.gz");
        NR_WARN("The file can not be saved as png and is converted to nifti " << filename << " -> " << fname);
        filename = fname.c_str();
        fileFormat = NR_NII_FORMAT;
    }

    // Convert the image to the correct format if required, set the filename and save the file
    switch (fileFormat) {
    case NR_NII_FORMAT:
        nifti_set_filenames(image, filename, 0, 0);
        nifti_image_write(image);
        break;
    case NR_PNG_FORMAT:
        reg_io_writePNGfile(image, filename);
        break;
#ifdef USE_NRRD
    case NR_NRRD_FORMAT:
        Nrrd *nrrdImage = reg_io_nifti2nrrd(image);
        reg_io_writeNRRDfile(nrrdImage, filename);
        nrrdNuke(nrrdImage);
        break;
#endif
    }
}
/* *************************************************************** */
template <class DataType>
void reg_io_displayImageData1(nifti_image *image) {
    NR_DEBUG("Image values:");
    const DataType *data = static_cast<DataType*>(image->data);
    const size_t nVoxelsPerVolume = NiftiImage::calcVoxelNumber(image, 3);

    size_t voxelIndex = 0;
    for (int z = 0; z < image->nz; z++) {
        for (int y = 0; y < image->ny; y++) {
            for (int x = 0; x < image->nx; x++) {
                std::string text = "[" + std::to_string(x) + " - " + std::to_string(y) + " - " + std::to_string(z) + "] = [";
                for (int tu = 0; tu < image->nt * image->nu; ++tu)
                    text += std::to_string(static_cast<double>(data[voxelIndex + tu * nVoxelsPerVolume])) + " ";
                if (text.back() == ' ')
                    text.pop_back();
                text += "]";
                NR_DEBUG(text);
            }
        }
    }
}
/* *************************************************************** */
void reg_io_displayImageData(nifti_image *image) {
    switch (image->datatype) {
    case NIFTI_TYPE_UINT8:
        reg_io_displayImageData1<unsigned char>(image);
        break;
    case NIFTI_TYPE_INT8:
        reg_io_displayImageData1<char>(image);
        break;
    case NIFTI_TYPE_UINT16:
        reg_io_displayImageData1<unsigned short>(image);
        break;
    case NIFTI_TYPE_INT16:
        reg_io_displayImageData1<short>(image);
        break;
    case NIFTI_TYPE_UINT32:
        reg_io_displayImageData1<unsigned>(image);
        break;
    case NIFTI_TYPE_INT32:
        reg_io_displayImageData1<int>(image);
        break;
    case NIFTI_TYPE_FLOAT32:
        reg_io_displayImageData1<float>(image);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_io_displayImageData1<double>(image);
        break;
    default:
        NR_FATAL_ERROR("Unsupported datatype");
    }
}
/* *************************************************************** */
