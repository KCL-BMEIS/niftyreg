%
% mindDescriptor_test('/home/bpresles/OneDriveBusiness/NiftyReg/refImg2D.nii.gz', ...
% '/home/bpresles/OneDriveBusiness/NiftyReg/refImg3D.nii.gz', './test-mind');
%
function [expectedMIND2DDescriptorImage_nii, expectedMIND3DDescriptorImage_nii] = ...
                            mindsscDescriptor_test(img2D, img3D, output_path,rescaleImg, mask2D, mask3D)
%
p=1;
%
%% 2D
%convUniformKernel=ones(2*p+1,2*p+1);
convKernel = fspecial('gaussian', 2*p+1, 0.5);
%%
refImg2D = load_untouch_nii(img2D); % read the Nifti file
RSampling = [+1 +1;+1 -1];
tx=[-1,+0,-1,+0];
ty=[+0,-1,+0,+1];
lengthDescriptor=4;
refImg2DImg = single(refImg2D.img);
%
if (nargin < 5)
    inputMask2D = ones(size(refImg2DImg));
else
    inputMask2D = mask2D;
end
%
idZeros=find(inputMask2D==0);
refImg2DImg(idZeros) = NaN;
%
if(rescaleImg)
    %% TO BE Consitent with NiftyReg - image rescaling
    minrefImg2D = double(min(refImg2DImg(:)));
    maxrefImg2D = double(max(refImg2DImg(:)));
    refImg2DImg = single((double(refImg2DImg)-minrefImg2D)./(maxrefImg2D-minrefImg2D));
    %refImg2DPrime = (refImg2DPrime-minrefImg2D)./(maxrefImg2D-minrefImg2D);
    %%
end
%
Dp_array = zeros([size(refImg2DImg) lengthDescriptor],'single');
Vp_array = zeros([size(refImg2DImg) lengthDescriptor],'single');
%
store_id=1;
%
for id=1:size(RSampling,2)
    %% Let's translate the image by rx, ry pixel
    rx=RSampling(1,id);
    ry=RSampling(2,id);
    idNaN = find(isnan(refImg2DImg));
    refImg2DImg(idNaN)=-999;
    refImg2DPrimeImg = imtranslate(refImg2DImg, [ry, rx], 'FillValues', 0); %NaN
    id999 = find(refImg2DImg==-999);
    refImg2DImg(id999)=NaN;
    id999 = find(refImg2DPrimeImg==-999);
    refImg2DPrimeImg(id999)=NaN;
    diffImg = single((double(refImg2DImg)-double(refImg2DPrimeImg)));
    diffImg = single(double(diffImg).^2);
    %% Have to correct the borders by hand
    maskImg = zeros(size(diffImg));%2*p+1-1
    maskImg(diffImg > -1) = 1;
    %inputMask2DT = imtranslate(inputMask2D, [ry, rx], 'FillValues', 0);
    maskImg=maskImg.*inputMask2D;
    %diffImgPadded = padarray(diffImg,[1 1],'circular');
    diffImg(isnan(diffImg))=0;
    diffImg=diffImg.*maskImg;
    imgConv = single(conv2(double(diffImg),convKernel,'same'));
    maskConv = single(conv2(double(maskImg),convKernel,'same'));
    imgConv = single(double(imgConv)./double(maskConv));
    imgConv(maskImg==0)=NaN;
    for idtr = 1:2
        idNaN = find(isnan(imgConv));
        imgConv(idNaN)=-999;
        imgConvPrime = imtranslate(imgConv, [ty(store_id), tx(store_id)], 'FillValues', 0);%NaN
        id999 = find(imgConv==-999);
        imgConv(id999)=NaN;
        id999 = find(imgConvPrime==-999);
        imgConvPrime(id999)=NaN;
        %%%%%%%%
        Dp_array(:,:,1,store_id) = imgConvPrime;
        Vp_array(:,:,store_id) = Dp_array(:,:,1,store_id);
        store_id = store_id +1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vp_image = mean(Vp_array,3);
idZeros=find(Vp_image==0);
Vp_image(idZeros)=eps;
%%
for id=1:lengthDescriptor
    MIND2D_descriptor(:,:,1,id) = single(exp(double(-Dp_array(:,:,1,id))./double(Vp_image)));
end
%% Normalise MIND max = 1
maxMind=max(MIND2D_descriptor,[],4);
%
for id=1:lengthDescriptor
    MIND2D_descriptor(:,:,1,id)=single(double(MIND2D_descriptor(:,:,1,id))./double(maxMind));
end
%% SAVE
% The floating and warped image should have the same datatype !
expectedMIND2DDescriptorImage_nii = make_nii(MIND2D_descriptor,...
    [refImg2D.hdr.dime.pixdim(2),...
    refImg2D.hdr.dime.pixdim(3),...
    refImg2D.hdr.dime.pixdim(4)],...
    [],...
    16); % 16 is float
%
save_nii(expectedMIND2DDescriptorImage_nii, [output_path,'/expectedMINDSSCDescriptor2D.nii.gz']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3D
convKernel = fGaussian3D(2*p+1, 0.5);
%%
refImg3D = load_untouch_nii(img3D); % read the Nifti file
RSampling = [+1 +1 -1 0 +1 0;+1 -1 0 -1 0 +1;0 0 +1 +1 +1 +1];
tx=[-1,+0,-1,+0,+0,+1,+0,+0,+0,-1,+0,+0];
ty=[+0,-1,+0,+1,+0,+0,+0,+1,+0,+0,+0,-1];
tz=[+0,+0,+0,+0,-1,+0,-1,+0,-1,+0,-1,+0];
lengthDescriptor=12;
refImg3DImg = single(refImg3D.img);
%
if (nargin < 6)
    inputMask3D = ones(size(refImg3DImg));
else
    inputMask3D = mask3D;
end
%
idZeros=find(inputMask3D==0);
refImg3DImg(idZeros) = NaN;
%
if(rescaleImg)
    %% TO BE Consitent with NiftyReg - image rescaling
    minrefImg3D = double(min(refImg3DImg(:)));
    maxrefImg3D = double(max(refImg3DImg(:)));
    refImg3DImg = single((double(refImg3DImg)-minrefImg3D)./(maxrefImg3D-minrefImg3D));
    %refImg3DPrime = (refImg3DPrime-minrefImg3D)./(maxrefImg3D-minrefImg3D);
    %%
end
%
Dp_array = zeros([size(refImg3D.img) lengthDescriptor],'single');
Vp_array = zeros([size(refImg3D.img) lengthDescriptor],'single');
%
store_id=1;
%
for id=1:size(RSampling,2)
    %% Let's translate the image by rx, ry pixel
    rx=RSampling(1,id);
    ry=RSampling(2,id);
    rz=RSampling(3,id);
    idNaN = find(isnan(refImg3DImg));
    refImg3DImg(idNaN)=-999;
    refImg3DPrimeImg = imtranslate(refImg3DImg, [ry, rx, rz], 'FillValues', 0);%NaN
    id999 = find(refImg3DImg==-999);
    refImg3DImg(id999)=NaN;
    id999 = find(refImg3DPrimeImg==-999);
    refImg3DPrimeImg(id999)=NaN;
    diffImg = single(double(refImg3DImg) - double(refImg3DPrimeImg));
    diffImg = single(double(diffImg).^2);
    %% Have to correct the borders by hand
    maskImg = zeros(size(diffImg));%2*p+1-1
    maskImg(diffImg > -1) = 1;
    %inputMask3DT = imtranslate(inputMask3D, [ry, rx, rz], 'FillValues', 0);
    maskImg=maskImg.*inputMask3D;
    %diffImgPadded = padarray(diffImg,[1 1],'circular');
    diffImg(isnan(diffImg))=0;
    diffImg=diffImg.*maskImg;
    imgConv = single(convn(double(diffImg),convKernel,'same'));
    maskConv = single(convn(double(maskImg),convKernel,'same'));
    imgConv = single(double(imgConv)./double(maskConv));
    imgConv(maskImg==0)=NaN;
    for idtr = 1:2
        idNaN = find(isnan(imgConv));
        imgConv(idNaN)=-999;
        imgConvPrime = imtranslate(imgConv, [ty(store_id), tx(store_id), tz(store_id)], 'FillValues', 0);%NaN
        id999 = find(imgConv==-999);
        imgConv(id999)=NaN;
        id999 = find(imgConvPrime==-999);
        imgConvPrime(id999)=NaN;
        %%%%%%%%
        Dp_array(:,:,:,store_id) = imgConvPrime;
        Vp_array(:,:,:,store_id) = Dp_array(:,:,:,store_id);
        store_id = store_id +1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vp_image = mean(Vp_array,4);
idZeros=find(Vp_image==0);
Vp_image(idZeros)=eps;
%%
for id=1:lengthDescriptor
    MIND3D_descriptor(:,:,:,id) = single(exp(double(-Dp_array(:,:,:,id))./double(Vp_image)));
end
%% Normalise MIND max = 1
maxMind=max(MIND3D_descriptor,[],4);
%max1=mean(mind,3);
for id=1:lengthDescriptor
    MIND3D_descriptor(:,:,:,id)=single(double(MIND3D_descriptor(:,:,:,id))./double(maxMind));
end
%% SAVE
% The floating and warped image should have the same datatype !
expectedMIND3DDescriptorImage_nii = make_nii(MIND3D_descriptor,...
    [refImg3D.hdr.dime.pixdim(2),...
    refImg3D.hdr.dime.pixdim(3),...
    refImg3D.hdr.dime.pixdim(4)],...
    [],...
    16); % 16 is float
%
save_nii(expectedMIND3DDescriptorImage_nii, [output_path,'/expectedMINDSSCDescriptor3D.nii.gz']);
end
