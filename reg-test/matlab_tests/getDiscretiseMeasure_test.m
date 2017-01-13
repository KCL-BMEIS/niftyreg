function discretisedValues = getDiscretiseMeasure_test(img3DRef, gridSpacing_vox, img3DWar, output_path, discrete_radius, discrete_increment)
currentRef = load_untouch_nii(img3DRef); % read the Nifti file
pixRefdim = currentRef.hdr.dime.pixdim(2:4);
gridSpacing_mm = gridSpacing_vox*pixRefdim;
currentRefImg = currentRef.img;
%% RESCALE THE IMAGE
minrefImg = double(min(currentRefImg(:)));
maxrefImg = double(max(currentRefImg(:)));
currentRefImg = single((double(currentRefImg)-minrefImg)./(maxrefImg-minrefImg));
%%
refImgSize=size(currentRefImg);
cpImgSize=floor(refImgSize/gridSpacing_vox + 4);
%%%
%currentRefImgPadded = padarray(currentRefImg,[gridSpacing_vox,gridSpacing_vox,gridSpacing_vox]);
%%%
%cpImage = load_untouch_nii(cpImage); % read the Nifti file
%cpImageImg = cpImage.img;
%%%
warImage = load_untouch_nii(img3DWar); % read the Nifti file
currentWarImg = warImage.img;
%% RESCALE THE IMAGE
minwarImg = double(min(currentWarImg(:)));
maxwarImg = double(max(currentWarImg(:)));
currentWarImg = single((double(currentWarImg)-minwarImg)./(maxwarImg-minwarImg));
%%
warImgSize=size(currentWarImg);
%%%
displacementArray=-discrete_radius:discrete_increment:discrete_radius;
nbDisplacement3D = length(displacementArray)^3;
%%%
%% FOR EACH CONTROL POINTS
coordCP_x = (1-1*gridSpacing_vox):gridSpacing_vox:(size(currentRefImg,1)+2*gridSpacing_vox);
coordCP_y = (1-1*gridSpacing_vox):gridSpacing_vox:(size(currentRefImg,2)+2*gridSpacing_vox);
coordCP_z = (1-1*gridSpacing_vox):gridSpacing_vox:(size(currentRefImg,3)+2*gridSpacing_vox);
nbControlPoints = length(coordCP_x) * length(coordCP_y) * length(coordCP_z);
%%%
blockSize=[gridSpacing_vox,gridSpacing_vox,gridSpacing_vox];
refBlockValues = zeros(1,blockSize(1)*blockSize(2)*blockSize(3));
warBlockValues = zeros(1,blockSize(1)*blockSize(2)*blockSize(3));
%%%
idOutput = 1;
discretisedValues = zeros(1,nbControlPoints*nbDisplacement3D);
%%%
for kk=1:length(coordCP_z)
    for jj=1:length(coordCP_y)
        for ii=1:length(coordCP_x)
            currentCP = [coordCP_x(ii), coordCP_y(jj), coordCP_z(kk)];
            currentCP_id = length(coordCP_y)*length(coordCP_x)*(kk-1)+length(coordCP_x)*(jj-1)+(ii-1);
            %% Lets retreive the reference block (zero padding)
            idRefBlock = 1;
            for bz=currentCP(3)-blockSize(3)/2:currentCP(3)+blockSize(3)/2-1
                for by=currentCP(2)-blockSize(2)/2:currentCP(2)+blockSize(2)/2-1
                    for bx=currentCP(1)-blockSize(1)/2:currentCP(1)+blockSize(1)/2-1
                        if(bx>=1 && bx <= refImgSize(1) && by>=1 && by <= refImgSize(2) && bz>=1 && bz <= refImgSize(3))
                            refBlockValues(idRefBlock) = currentRefImg(bx,by,bz);
                            if(isnan(refBlockValues(idRefBlock)))
                                refBlockValues(idRefBlock) = 0;
                            end
                        else
                            refBlockValues(idRefBlock) = 0;
                        end
                        idRefBlock = idRefBlock + 1;
                    end
                end
            end
            %
            %% FOR ALL displacement
            for dz=1:length(displacementArray)
                for dy=1:length(displacementArray)
                    for dx=1:length(displacementArray)
                        %% NEW POSITION of the control points
                        newPositionCP = [currentCP(1) + displacementArray(dx), currentCP(2) + displacementArray(dy), currentCP(3) + displacementArray(dz)];
                        idWarBlock = 1;
                        
                        for bz=newPositionCP(3)-blockSize(3)/2:newPositionCP(3)+blockSize(3)/2-1
                            for by=newPositionCP(2)-blockSize(2)/2:newPositionCP(2)+blockSize(2)/2-1
                                for bx=newPositionCP(1)-blockSize(1)/2:newPositionCP(1)+blockSize(1)/2-1
                                    if(bx>=1 && bx <= warImgSize(1) && by>=1 && by <= warImgSize(2) && bz>=1 && bz <= warImgSize(3))
                                        warBlockValues(idWarBlock) = currentWarImg(bx,by,bz);
                                        if(isnan(warBlockValues(idWarBlock)))
                                            warBlockValues(idWarBlock) = 0;
                                        end
                                    else
                                        warBlockValues(idWarBlock) = 0;
                                    end
                                    idWarBlock = idWarBlock + 1;
                                end
                            end
                        end
                        %% COMPUTE THE SIMILARITY MEASURE:
                        %currentCP_id
                        simMeasure=sum((refBlockValues-warBlockValues).^2);
                        if(isnan(simMeasure))
                            disp('NaN');
                        end
                        discretisedValues(idOutput)=simMeasure;
                        idOutput = idOutput+1;
                        %
                    end
                end
            end
        end
    end
end
%% SAVE THE RESULTS
save([output_path,'/discretisedValues.mat'],'discretisedValues');
fileID = fopen([output_path,'/discretisedValues.dat'],'w');
fwrite(fileID,discretisedValues,'single');
fclose(fileID);
%%
