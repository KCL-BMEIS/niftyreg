function MINDSSD_test(img2D1, img2D2, img3D1, img3D2, output_path)
%% CREATE A MASK FIRST for 2D and 3D !!!!
img2D1Img = load_untouch_nii(img2D1); % read the Nifti file
img2D1Img = img2D1Img.img;
img2D2Img = load_untouch_nii(img2D2); % read the Nifti file
img2D2Img = img2D2Img.img;

%% RESCALE THE IMAGES FIRST !!!
min2D1Img = double(min(img2D1Img(:)));
max2D1Img = double(max(img2D1Img(:)));
img2D1Img = single((double(img2D1Img)-min2D1Img)./(max2D1Img-min2D1Img));

min2D2Img = double(min(img2D2Img(:)));
max2D2Img = double(max(img2D2Img(:)));
img2D2Img = single((double(img2D2Img)-min2D2Img)./(max2D2Img-min2D2Img));
%%
combinedMask2D1 = zeros(size(img2D1Img)); %img2D2Img should have the same size
combinedMask2D2 = zeros(size(img2D2Img)); %img2D2Img should have the same size
combinedMask2D1(img2D1Img > -1) = 1;
combinedMask2D2(img2D2Img > -1) = 1;
combinedMask2D = combinedMask2D1.*combinedMask2D2;
%% 3D
img3D1Img = load_untouch_nii(img3D1); % read the Nifti file
img3D1Img = img3D1Img.img;
img3D2Img = load_untouch_nii(img3D2); % read the Nifti file
img3D2Img = img3D2Img.img;

%% RESCALE THE IMAGES FIRST !!!
min3D1Img = double(min(img3D1Img(:)));
max3D1Img = double(max(img3D1Img(:)));
img3D1Img = single((double(img3D1Img)-min3D1Img)./(max3D1Img-min3D1Img));

min3D2Img = double(min(img3D2Img(:)));
max3D2Img = double(max(img3D2Img(:)));
img3D2Img = single((double(img3D2Img)-min3D2Img)./(max3D2Img-min3D2Img));

combinedMask3D1 = zeros(size(img3D1Img)); %img2D2Img should have the same size
combinedMask3D2 = zeros(size(img3D2Img)); %img2D2Img should have the same size
combinedMask3D1(img3D1Img > -1) = 1;
combinedMask3D2(img3D2Img > -1) = 1;
combinedMask3D = combinedMask3D1.*combinedMask3D2;
%% FIRST MIND DESCRIPTOR !!!!
[expectedMIND2DDescriptorImage_nii1, expectedMIND3DDescriptorImage_nii1] = ...
    mindDescriptor_test(img2D1, img3D1, output_path,true,combinedMask2D,combinedMask3D);
delete([output_path,'/expectedMINDDescriptor2D.nii.gz']);
delete([output_path,'/expectedMINDDescriptor3D.nii.gz']);
[expectedMIND2DDescriptorImage_nii2, expectedMIND3DDescriptorImage_nii2] = ...
    mindDescriptor_test(img2D2, img3D2, output_path,true,combinedMask2D,combinedMask3D);
delete([output_path,'/expectedMINDDescriptor2D.nii.gz']);
delete([output_path,'/expectedMINDDescriptor3D.nii.gz']);
%%
MINDArray = [expectedMIND2DDescriptorImage_nii1, expectedMIND2DDescriptorImage_nii2;...
             expectedMIND3DDescriptorImage_nii1, expectedMIND3DDescriptorImage_nii2];
%%
for ii=1:size(MINDArray,1)
    current1=MINDArray(ii,1);
    current2=MINDArray(ii,2);
    currentImg1 = double(current1.img);
    currentImg2 = double(current2.img);
    %
    dimVect1 = current1.hdr.dime.dim;
    dimVect2 = current2.hdr.dime.dim;
    %
    nz1 = dimVect1(4);
    nz2 = dimVect2(4);
    if nz1 ~= nz2
        error('Error. Images must have the same dimension.');
    end
    if nz1 > 1
        imgDim = 3;
    else
        imgDim = 2;
    end
    %
    nt1 = dimVect1(5);
    nt2 = dimVect2(5);
    %
    if (nt1 ~= nt2)
        error('Error. Images must have the same dimension.');
    end
    %
    SSDValue = 0;
    for jj=1:nt1
        img1 = currentImg1(:,:,:,jj);
        img2 = currentImg2(:,:,:,jj);
        diff2 = (img1-img2).^2;
        diff2Array=diff2(:);
        SSDValue = SSDValue + sum(diff2Array(~isnan(diff2Array)))/length(diff2Array(~isnan(diff2Array)));
    end
    %% Write the result
    expectedSSD = -SSDValue;%/nt1;
    dlmwrite([output_path,'/expectedMINDSSDValue',num2str(imgDim),'D.txt'],expectedSSD,'precision','%.6f','delimiter',' ');
end
end