#include "_reg_maths.h"
#include "nifti1_io.h"
#define PI 3.14159265

int main ()
{
    mat44 I;
    I.m[0][0]=0.962807;
    I.m[0][1]=-0.320115;
    I.m[0][2]=0.339893;
    I.m[0][3]=0;
    I.m[1][0]=-0.320115;
    I.m[1][1]=1.48703;
    I.m[1][2]=-0.440168;
    I.m[1][3]=0;
    I.m[2][0]=0.339893;
    I.m[2][1]=-0.440168;
    I.m[2][2]=1.36216;
    I.m[2][3]=0;
    I.m[3][0]=0;
    I.m[3][1]=0;
    I.m[3][2]=0;
    I.m[3][3]=1;
    reg_mat44_disp(&I,(char*)"*Single* - Original");

    mat44 sqrtI=reg_mat44_sqrt(&I);
    reg_mat44_disp(&sqrtI,(char*)"*Single* - Square root of Original");

    mat44 invI=reg_mat44_inv(&I);
    reg_mat44_disp(&invI,(char*)"*Single* - invert of Original");

    mat44 logI=reg_mat44_logm(&I);
    reg_mat44_disp(&logI,(char*)"*Single* - Log of original");

    mat44 expI=reg_mat44_expm(&I);
    reg_mat44_disp(&expI,(char*)"*Single* - Exponential of original");

    mat44 Ilog=reg_mat44_expm(&logI);
    reg_mat44_disp(&Ilog,(char*)"*Single* - Log-Exp of Original");

    mat44 Iexp=reg_mat44_logm(&expI);
    reg_mat44_disp(&Iexp,(char*)"*Single* - Exp-Log of Original");

    return 0;
}
