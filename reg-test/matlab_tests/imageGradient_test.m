%% COMPUTE GRADIENT
function imageGradient_test(img2D, img3D, output_path)
imgCell={img2D img3D};
for img=1:length(imgCell)
    % Read the nifti file
    current = load_untouch_nii(imgCell{img});
    % Define the input image dim
    inDimVect = current.hdr.dime.dim;
    if inDimVect(4) > 1
        imgDim = 3;
    else
        imgDim = 2;
    end
    % Define the padded image dim
    padDimVect = inDimVect;
    padDimVect(2) = padDimVect(2) + 2;
    padDimVect(3) = padDimVect(3) + 2;
    if imgDim > 2
        padDimVect(4) = padDimVect(4) + 2;
    end
    % pad the input image with zeros
    currentImgnD = zeros(padDimVect(2:end));
    if imgDim > 2
        currentImgnD(2:end-1,2:end-1,2:end-1,:) = current.img;
    else
        currentImgnD(2:end-1,2:end-1,:,:) = current.img;
    end
    % Define the kernel to use
    if imgDim == 2
        convKernelX = [1 0 -1]/2;
        convKernelY = convKernelX';
    else
        convKernelX = [1 0 -1]/2;
        convKernelY = convKernelX';
        convKernelZ(1,1,1) = 1/2;
        convKernelZ(1,1,2) = 0;
        convKernelZ(1,1,3)= -1/2;
    end
    % Create an image to save the gradient
    sizeImgGrad = [size(currentImgnD),imgDim];
    imgGradient = zeros(sizeImgGrad);
    % Iterate over the time points
    for ii=1:inDimVect(5)
        currentImg=currentImgnD(:,:,:,ii);
        % Convolution of the x axis
        imgGradientX = single(convn(double(currentImg),convKernelX,'valid'));
        imgGradientX(isnan(imgGradientX))=0;
        % Convolution of the y axis
        imgGradientY = single(convn(double(currentImg),convKernelY,'valid'));
        imgGradientY(isnan(imgGradientY))=0;
        % Convolution of the z axis
        if imgDim == 3
            imgGradientZ = single(convn(double(currentImg),convKernelZ,'valid'));
            imgGradientZ(isnan(imgGradientZ))=0;
        end
        %
        if imgDim == 2
            %
            imgGradient(2:end-1,:,1,ii,1)=imgGradientY;
            imgGradient(:,2:end-1,1,ii,2)=imgGradientX;
        else
            %
            imgGradient(2:end-1,:,:,ii,1)=imgGradientY;
            imgGradient(:,2:end-1,:,ii,2)=imgGradientX;
            imgGradient(:,:,2:end-1,ii,3)=imgGradientZ;
        end
    end
    %% SAVE
    % The floating and warped image should have the same datatype !
    if imgDim > 2
        gradient_nii = make_nii(imgGradient(2:end-1,2:end-1,2:end-1,:),...
            [current.hdr.dime.pixdim(2),...
            current.hdr.dime.pixdim(3),...
            current.hdr.dime.pixdim(4)],...
            [],...
            16); % 16 is float
    else
        gradient_nii = make_nii(imgGradient(2:end-1,2:end-1,:,:),...
            [current.hdr.dime.pixdim(2),...
            current.hdr.dime.pixdim(3),...
            current.hdr.dime.pixdim(4)],...
            [],...
            16); % 16 is float
    end
    %
    save_nii(gradient_nii, ...
        [output_path,'/expectedImageGradient',num2str(imgDim),'D.nii.gz']);
end
end