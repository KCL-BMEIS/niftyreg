function changeDataType_test(refImg2D_name, refImg3D_name, output_path)
%%
%% Change image datatype tests
%%
double_datatype = 64;
double_bitpix = 64;
float_datatype = 16;
float_bitpix = 32;
uchar_datatype = 2;
uchar_bitpix = 8;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOUBLE
refImg3D=load_untouch_nii(refImg3D_name);
%
double_nii = refImg3D;
double_nii.img = double(refImg3D.img);
double_nii.hdr.dime.datatype = double_datatype;
double_nii.hdr.dime.bitpix = double_bitpix;
%
save_untouch_nii(double_nii, [output_path,'/refImg3D_double.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SINGLE
refImg3D=load_untouch_nii(refImg3D_name);
%
single_nii = refImg3D;
single_nii.img = single(refImg3D.img);
single_nii.hdr.dime.datatype = float_datatype;
single_nii.hdr.dime.bitpix = float_bitpix;
%
save_untouch_nii(single_nii, [output_path,'/refImg3D_float.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UCHAR
refImg3D=load_untouch_nii(refImg3D_name);
%
uchar_nii = refImg3D;
uchar_nii.img = uint8(refImg3D.img);
uchar_nii.hdr.dime.datatype = uchar_datatype;
uchar_nii.hdr.dime.bitpix = uchar_bitpix;
%
save_untouch_nii(uchar_nii, [output_path,'/refImg3D_uchar.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DOUBLE
refImg2D=load_untouch_nii(refImg2D_name);
%
double_nii = refImg2D;
double_nii.img = double(refImg2D.img);
double_nii.hdr.dime.datatype = double_datatype;
double_nii.hdr.dime.bitpix = double_bitpix;
%
save_untouch_nii(double_nii, [output_path,'/refImg2D_double.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SINGLE
refImg2D=load_untouch_nii(refImg2D_name);
%
single_nii = refImg2D;
single_nii.img = single(refImg2D.img);
single_nii.hdr.dime.datatype = float_datatype;
single_nii.hdr.dime.bitpix = float_bitpix;
%
save_untouch_nii(single_nii, [output_path,'/refImg2D_float.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UCHAR
refImg2D=load_untouch_nii(refImg2D_name);
%
uchar_nii = refImg2D;
uchar_nii.img = uint8(refImg2D.img);
uchar_nii.hdr.dime.datatype = uchar_datatype;
uchar_nii.hdr.dime.bitpix = uchar_bitpix;
%
save_untouch_nii(uchar_nii, [output_path,'/refImg2D_uchar.nii.gz']);
end