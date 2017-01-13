function affineDeformationField_test(refImg2D_name, refImg3D_name, output_path)
%% cout float  3.333333253860474
%% cout double 3.333333333333334
%% The number of digits of precision a value has depends on both the size 
%% (floats have less precision than doubles) and the particular value being stored 
%% (some values have more precision than others).
%% Float values have between 6 and 9 digits of precision,
%% with most float values having at least 7 significant digits
%% (which is why everything after that many digits in our answer above is junk).
%% Double values have between 15 and 18 digits of precision,
%% with most double values having at least 16 significant digits.
%% Long double has a minimum precision of 15, 18, or 33 significant digits
%% depending on how many bytes it occupies.
%%
%% Create a vector field image from an affine transformation matrix
%%
double_datatype = 64;
double_bitpix = 64;
float_datatype = 16;
float_bitpix = 32;
uchar_datatype = 2;
uchar_bitpix = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
refImg3D = load_untouch_nii(refImg3D_name); % read the Nifti file
HMatrix = eye(4,4,'single');
HMatrix(1,:) = refImg3D.hdr.hist.srow_x;
HMatrix(2,:) = refImg3D.hdr.hist.srow_y;
HMatrix(3,:) = refImg3D.hdr.hist.srow_z;
%HMatrix
%% Generate a random affine tansformation matrix:
%A=rand(4,4,'single');
A=rand(4,4)+ eye(4,4);
A(end,:)  =[0 0 0 1];
A=single(A);
%%
% A(1,:)=[0.9074931	0.01773963	-0.01136361	-1.596654];
% A(2,:)=[0.009297408	0.9127377	-0.1370281	-7.986477];
% A(3,:)=[0.01029972	0.1200682	0.8102149	-0.5980158];
% A(4,:)=[0	0	0	1];
% A
dlmwrite([output_path,'/affine_mat3D.txt'],A,'precision','%.6f','delimiter',' ');
%% READ THE FILE THAT WE HAVE JUST WRITTEN
fileID = fopen([output_path,'/affine_mat3D.txt'],'r');
formatSpec = '%f';
sizeA = [4 4];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
A=single(A');
%%
expectedField = zeros(refImg3D.hdr.dime.dim(2),refImg3D.hdr.dime.dim(3),refImg3D.hdr.dime.dim(4),1,3,'single');
%%
globalMatrix = single(double(A) * double(HMatrix));
%%
for kk=1:size(refImg3D.img,3)
    for jj=1:size(refImg3D.img,2)
        for ii=1:size(refImg3D.img,1)
            newPosition = single(double(globalMatrix) * double([ii-1 jj-1 kk-1 1]'));
            expectedField(ii,jj,kk,1,1)=newPosition(1);
            expectedField(ii,jj,kk,1,2)=newPosition(2);
            expectedField(ii,jj,kk,1,3)=newPosition(3);
        end
    end
end
%% Usage: nii = make_nii(img, [voxel_size], [origin], [datatype], [description])
%% 16 = float32
%% 64 = float64
expectedField_nii = make_nii(expectedField,...
    [refImg3D.hdr.dime.pixdim(2),...
    refImg3D.hdr.dime.pixdim(3),...
    refImg3D.hdr.dime.pixdim(4)],...
    [],float_datatype);
%
save_nii(expectedField_nii, [output_path,'/affine_def3D.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
refImg2D = load_untouch_nii(refImg2D_name); % read the Nifti file
HMatrix = eye(4,4,'single');
HMatrix(1,:) = refImg2D.hdr.hist.srow_x;
HMatrix(2,:) = refImg2D.hdr.hist.srow_y;
HMatrix(3,:) = refImg2D.hdr.hist.srow_z;
%HMatrix
%% Generate a random affine tansformation matrix:
%A=rand(4,4,'single');
A=rand(4,4) + eye(4,4);
A(1,3)=0;
A(2,3)=0;
A(end-1,:) = [0 0 1 0];
A(end,:) = [0 0 0 1];
A=single(A);
%%
% A(1,:)=[1.032482 -0.007169947 0 -0.6956705];
% A(2,:)=[0.07810403 1.005479 0 0.01127497];
% A(3,:)=[0 0 1 0];
% A(4,:)=[0	0 0	1];
% A
dlmwrite([output_path,'/affine_mat2D.txt'],A,'precision','%.6f','delimiter',' ');
%% READ THE FILE THAT WE HAVE JUST WRITTEN
fileID = fopen([output_path,'/affine_mat2D.txt'],'r');
formatSpec = '%f';
sizeA = [4 4];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
A=single(A');
%%
expectedField = zeros(refImg2D.hdr.dime.dim(2),refImg2D.hdr.dime.dim(3),refImg2D.hdr.dime.dim(4),1,2,'single');
%%
globalMatrix = single(double(A) * double(HMatrix));
%%
for kk=1:size(refImg2D.img,3)
    for jj=1:size(refImg2D.img,2)
        for ii=1:size(refImg2D.img,1)
            newPosition = single(double(globalMatrix) * double([ii-1 jj-1 kk-1 1]'));
            expectedField(ii,jj,kk,1,1)=newPosition(1);
            expectedField(ii,jj,kk,1,2)=newPosition(2);
        end
    end
end
%% Usage: nii = make_nii(img, [voxel_size], [origin], [datatype], [description])
%% 16 = float32
%% 64 = float64
expectedField_nii = make_nii(expectedField,...
    [refImg2D.hdr.dime.pixdim(2),...
    refImg2D.hdr.dime.pixdim(3),...
    refImg2D.hdr.dime.pixdim(4)],...
    [],float_datatype);
%
save_nii(expectedField_nii, [output_path,'/affine_def2D.nii.gz']);