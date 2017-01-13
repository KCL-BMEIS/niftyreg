function SSD_test(img1, img2, output_path)
    current1 = load_untouch_nii(img1); % read the Nifti file
    current2 = load_untouch_nii(img2); % read the Nifti file
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
    for ii=1:nt1
        img1 = currentImg1(:,:,:,ii);
        img2 = currentImg2(:,:,:,ii);
        minImg1 = min(img1(:));
        maxImg1 = max(img1(:));
        minImg2 = min(img2(:));
        maxImg2 = max(img2(:));
        img1 = (img1-minImg1)./(maxImg1-minImg1);
        img2 = (img2-minImg2)./(maxImg2-minImg2);
        diff2 = (img1-img2).^2;
        diff2Array=diff2(:);
        SSDValue = SSDValue + sum(diff2Array(~isnan(diff2Array)))/sum(~isnan(diff2Array));
    end
    %% Write the result
    expectedSSD = -SSDValue;
    dlmwrite([output_path,'/expectedSSDValue',num2str(imgDim),'D.txt'],expectedSSD,'precision','%.6f','delimiter',' ');
end