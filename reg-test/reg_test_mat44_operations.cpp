#include "_reg_maths.h"
#include "nifti1_io.h"
#define PI 3.14159265

int main ()
{
    printf("****************************\n");
    printf("Tests using single precision\n");
    printf("****************************\n");

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

    printf("\n****************************\n");
    printf("Tests using double precision\n");
    printf("****************************\n");
    reg_mat44d I_d = reg_mat44_singleToDouble(&I);
    reg_mat44_disp(&I_d,(char*)"*Double* - Original");

    reg_mat44d sqrtI_d=reg_mat44_sqrt(&I_d);
    reg_mat44_disp(&sqrtI_d,(char*)"*Double* - Square root of Original");

    reg_mat44d invI_d=reg_mat44_inv(&I_d);
    reg_mat44_disp(&invI_d,(char*)"*Double* - invert of Original");

    reg_mat44d logI_d=reg_mat44_logm(&I_d);
    reg_mat44_disp(&logI_d,(char*)"*Double* - Log of original");

    reg_mat44d expI_d=reg_mat44_expm(&I_d);
    reg_mat44_disp(&expI_d,(char*)"*Double* - Exponential of original");

    reg_mat44d Ilog_d=reg_mat44_expm(&logI_d);
    reg_mat44_disp(&Ilog_d,(char*)"*Double* - Log-Exp of Original");

    reg_mat44d Iexp_d=reg_mat44_logm(&expI_d);
    reg_mat44_disp(&Iexp_d,(char*)"*Double* - Exp-Log of Original");

    return 0;
}
