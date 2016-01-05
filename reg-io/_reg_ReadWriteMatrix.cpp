#include "_reg_ReadWriteMatrix.h"
#include "_reg_maths.h"
//STD
#include <string>
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_ReadAffineFile(mat44 *mat,
                             nifti_image *target,
                             nifti_image *source,
                             char *fileName,
                             bool flirtFile)
{
    std::ifstream affineFile;
    affineFile.open(fileName);
    if(affineFile.is_open())
    {
        int i=0;
        double value1,value2,value3,value4;
        while(!affineFile.eof())
        {
            affineFile >> value1 >> value2 >> value3 >> value4;
            mat->m[i][0] = (float) value1;
            mat->m[i][1] = (float) value2;
            mat->m[i][2] = (float) value3;
            mat->m[i][3] = (float) value4;
            i++;
            if(i>3) break;
        }
    }
    else
    {
        char text[255];sprintf(text, "The affine file can not be read: %s", fileName);
        reg_print_fct_error("reg_tool_ReadAffineFile");
        reg_print_msg_error(text);
        reg_exit();
    }
    affineFile.close();

#ifndef NDEBUG
    reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] Read affine transformation");
#endif

    if(flirtFile)
    {
        mat44 absoluteTarget;
        mat44 absoluteSource;
        for(int i=0; i<4; i++)
        {
            for(int j=0; j<4; j++)
            {
                absoluteTarget.m[i][j]=absoluteSource.m[i][j]=0.0;
            }
        }
        //If the target sform is defined, it is used; qform otherwise;
        mat44 *targetMatrix;
        if(target->sform_code > 0)
        {
            targetMatrix = &(target->sto_xyz);
#ifndef NDEBUG
            reg_print_msg_debug("The target sform matrix is defined and used");
#endif
        }
        else targetMatrix = &(target->qto_xyz);
        //If the source sform is defined, it is used; qform otherwise;
        mat44 *sourceMatrix;
        if(source->sform_code > 0)
        {
#ifndef NDEBUG
            reg_print_msg_debug(" The source sform matrix is defined and used");
#endif
            sourceMatrix = &(source->sto_xyz);
        }
        else sourceMatrix = &(source->qto_xyz);

        for(int i=0; i<3; i++)
        {
            absoluteTarget.m[i][i]=sqrt(targetMatrix->m[0][i]*targetMatrix->m[0][i]
                    + targetMatrix->m[1][i]*targetMatrix->m[1][i]
                    + targetMatrix->m[2][i]*targetMatrix->m[2][i]);
            absoluteSource.m[i][i]=sqrt(sourceMatrix->m[0][i]*sourceMatrix->m[0][i]
                    + sourceMatrix->m[1][i]*sourceMatrix->m[1][i]
                    + sourceMatrix->m[2][i]*sourceMatrix->m[2][i]);
        }
        absoluteTarget.m[3][3]=absoluteSource.m[3][3]=1.0;
#ifndef NDEBUG
        reg_print_msg_debug("An flirt affine file is assumed and is converted to a real word affine matrix");
        reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] Matrix read from the input file");
        reg_mat44_disp(targetMatrix, (char *)"[NiftyReg DEBUG] Target Matrix");
        reg_mat44_disp(sourceMatrix, (char *)"[NiftyReg DEBUG] Source Matrix");
        reg_mat44_disp(&(absoluteTarget), (char *)"[NiftyReg DEBUG] Target absolute Matrix");
        reg_mat44_disp(&(absoluteSource), (char *)"[NiftyReg DEBUG] Source absolute Matrix");
#endif

        absoluteSource = nifti_mat44_inverse(absoluteSource);
        *mat = nifti_mat44_inverse(*mat);

        *mat = reg_mat44_mul(&absoluteSource,mat);
        *mat = reg_mat44_mul(mat, &absoluteTarget);
        *mat = reg_mat44_mul(sourceMatrix,mat);
        mat44 tmp = nifti_mat44_inverse(*targetMatrix);
        *mat = reg_mat44_mul(mat, &tmp);
    }

#ifndef NDEBUG
    reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] Affine matrix");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_ReadAffineFile(mat44 *mat,
                             char *fileName)
{
    std::ifstream affineFile;
    affineFile.open(fileName);
    if(affineFile.is_open())
    {
        int i=0;
        double value1,value2,value3,value4;
#ifndef NDEBUG
        char text_header[255];
        sprintf(text_header, "Affine matrix values:");
        reg_print_msg_debug(text_header);
#endif
        while(!affineFile.eof())
        {
            affineFile >> value1 >> value2 >> value3 >> value4;
#ifndef NDEBUG
            char text[255];
            sprintf(text, "%f - %f - %f - %f", value1, value2, value3, value4);
            reg_print_msg_debug(text);
#endif
            mat->m[i][0] = (float) value1;
            mat->m[i][1] = (float) value2;
            mat->m[i][2] = (float) value3;
            mat->m[i][3] = (float) value4;
            i++;
            if(i>3) break;
        }
    }
    else
    {
        char text[255];sprintf(text, "The affine file can not be read: %s", fileName);
        reg_print_fct_error("reg_tool_ReadAffineFile");
        reg_print_msg_error(text);
        reg_exit();
    }
    affineFile.close();

#ifndef NDEBUG
    reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] Affine matrix");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_WriteAffineFile(mat44 *mat,
                              const char *fileName)
{
    FILE *affineFile;
    affineFile=fopen(fileName, "w");
    for(int i=0; i<4; i++)
        fprintf(affineFile, "%.7g %.7g %.7g %.7g\n", mat->m[i][0], mat->m[i][1], mat->m[i][2], mat->m[i][3]);
    fclose(affineFile);
}
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
std::pair<size_t, size_t> reg_tool_sizeInputMatrixFile(char *filename)
{
    //FIRST LET'S DETERMINE THE NUMBER OF LINE AND COLUMN
    std::string line;
    std::ifstream matrixFile(filename);
    size_t nbLine = 0;
    size_t nbColumn = 0;
    if (matrixFile.is_open()) {
        std::getline(matrixFile, line);
        nbLine++;
        std::string delimiter = " ";
        size_t pos = 0;
        std::string token;
        //
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            nbColumn++;
            line.erase(0, pos + delimiter.length());
        }
        nbColumn++;
        //
        while (std::getline(matrixFile, line)) {
            nbLine++;
        }
        //
        matrixFile.close();
    }
    else {
        char text[255];
        sprintf(text, "The file can not be read: %s", filename);
        reg_print_fct_error("reg_tool_ReadMatrixFile");
        reg_print_msg_error(text);
        reg_exit();
    }
    std::pair <size_t, size_t> result(nbLine, nbColumn);
    return result;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T** reg_tool_ReadMatrixFile(char *filename, size_t nbLine, size_t nbColumn)
{
    //THEN CONSTRUCT THE MATRIX
    // Allocate the matrices
    T** mat = reg_matrix2DAllocate<T>(nbLine, nbColumn);
    //STORE THE VALUES
    std::string line;
    std::ifstream matrixFile(filename);
    double currentValue = 0;
    if (matrixFile.is_open()) {
        int j = 0;
        while (std::getline(matrixFile, line))
        {
            std::string delimiter = " ";
            int i = 0;
            size_t pos = 0;
            std::string token;
            while ((pos = line.find(delimiter)) != std::string::npos)
            {
                token = line.substr(0, pos);
                currentValue = atof(token.c_str());
                mat[j][i] = currentValue;
                line.erase(0, pos + delimiter.length());
                i++;
            }
            currentValue = atof(line.c_str());
            mat[j][i] = currentValue;
            j++;
        }
        matrixFile.close();
    }
    else
    {
        char text[255];
        sprintf(text, "The matrix file can not be read: %s", filename);
        reg_print_fct_error("reg_tool_ReadMatrixFile");
        reg_print_msg_error(text);
        reg_exit();
    }
    //
    return mat;
}
template float** reg_tool_ReadMatrixFile<float>(char *filename, size_t nbLine, size_t nbColumn);
template double** reg_tool_ReadMatrixFile<double>(char *filename, size_t nbLine, size_t nbColumn);
/* *************************************************************** */
/* *************************************************************** */
mat44* reg_tool_ReadMat44File(char *fileName)
{
    mat44 *mat = (mat44 *)malloc(sizeof(mat44));
    std::ifstream matrixFile;
    matrixFile.open(fileName);
    if (matrixFile.is_open()) {
        int i = 0;
        double value1, value2, value3, value4;
        while (!matrixFile.eof()) {
            matrixFile >> value1 >> value2 >> value3 >> value4;

            mat->m[i][0] = (float) value1;
            mat->m[i][1] = (float) value2;
            mat->m[i][2] = (float) value3;
            mat->m[i][3] = (float) value4;
            i++;
            if (i>3) break;
        }
    }
    else {
        char text[255]; sprintf(text, "The mat44 file can not be read: %s", fileName);
        reg_print_fct_error("reg_tool_ReadMat44File");
        reg_print_msg_error(text);
        reg_exit();
    }
    matrixFile.close();

#ifndef NDEBUG
    reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] mat44 matrix");
#endif

    return mat;
}
/* *************************************************************** */
/* *************************************************************** */
