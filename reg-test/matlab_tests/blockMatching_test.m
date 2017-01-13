function blockMatching_test(refImg2D_name, refImg3D_name, output_path)
indexImage=1;
vectImageName = {refImg2D_name, refImg3D_name};
%%2D and 3D version
for dim=[2,3]
    BLOCK_WIDTH = 4;
    BLOCK_WIDTH_MINUS1 = BLOCK_WIDTH-1;
    refImg=load_untouch_nii(vectImageName{indexImage});
    HMatrix = eye(4,4,'single');
    HMatrix(1,:) = refImg.hdr.hist.srow_x;
    HMatrix(2,:) = refImg.hdr.hist.srow_y;
    HMatrix(3,:) = refImg.hdr.hist.srow_z;
    refImgImg = refImg.img;
    %% Size of the input image:
    [m,n,o]=size(refImgImg);
    %%
    floImg=load_untouch_nii(vectImageName{indexImage});
    floImgImg = floImg.img;
    expectedBlockMatching = zeros(0,6);
    expectedBlockMatchingPixel = zeros(0,6);
    %% Let's cut our image in BLOCK_WIDTH*BLOCK_WIDTH size
    nbTotalBlock_m = ceil(m/BLOCK_WIDTH);
    nbTotalBlock_n = ceil(n/BLOCK_WIDTH);
    nbTotalBlock_o = ceil(o/BLOCK_WIDTH);
    if dim==2
        nMax_o=2;
    else
        nMax_o=nbTotalBlock_o-1;
    end
    %% Let's move the middle blocks
    %[-5,5]. -5 + (5+5)*rand(10,1);
    blockIndex=1;
    for oo=2:3:nMax_o
        for nn=2:3:nbTotalBlock_n-1
            for mm=2:3:nbTotalBlock_m-1
                if dim==2
                    blockCoord_mno=[(1+(mm-1)*BLOCK_WIDTH),(1+(nn-1)*BLOCK_WIDTH),1];
                else
                    blockCoord_mno=[(1+(mm-1)*BLOCK_WIDTH),(1+(nn-1)*BLOCK_WIDTH),(1+(oo-1)*BLOCK_WIDTH)];
                end
                
                blockCoord_xyz=[blockCoord_mno(1)-1, blockCoord_mno(2)-1, blockCoord_mno(3)-1];
                
                if dim==2
                    if(blockCoord_mno(1) >= 1 && (blockCoord_mno(1)+BLOCK_WIDTH_MINUS1) <=m...
                            && blockCoord_mno(2) >= 1 && (blockCoord_mno(2)+BLOCK_WIDTH_MINUS1) <=n)
                        
                        tmpImg = refImgImg(blockCoord_mno(1):blockCoord_mno(1)+BLOCK_WIDTH_MINUS1,...
                            blockCoord_mno(2):blockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                        
                    else
                        mEnd = min(m,blockCoord_mno(1)+BLOCK_WIDTH_MINUS1);
                        nEnd = min(n,blockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                        
                        tmpImg = refImgImg(blockCoord_mno(1):mEnd,blockCoord_mno(2):nEnd);
                        
                    end
                elseif dim==3
                    if(blockCoord_mno(1) >= 1 && (blockCoord_mno(1)+BLOCK_WIDTH_MINUS1) <=m...
                            && blockCoord_mno(2) >= 1 && (blockCoord_mno(2)+BLOCK_WIDTH_MINUS1) <=n...
                            && blockCoord_mno(3) >= 1 && (blockCoord_mno(3)+BLOCK_WIDTH_MINUS1) <=o)
                        
                        tmpImg = refImgImg(blockCoord_mno(1):blockCoord_mno(1)+BLOCK_WIDTH_MINUS1,...
                            blockCoord_mno(2):blockCoord_mno(2)+BLOCK_WIDTH_MINUS1,...
                            blockCoord_mno(3):blockCoord_mno(3)+BLOCK_WIDTH_MINUS1);
                        
                    else
                        mEnd = min(m,blockCoord_mno(1)+BLOCK_WIDTH_MINUS1);
                        nEnd = min(n,blockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                        oEnd = min(o,blockCoord_mno(3)+BLOCK_WIDTH_MINUS1);
                        
                        tmpImg = refImgImg(blockCoord_mno(1):mEnd,...
                            blockCoord_mno(2):nEnd,...
                            blockCoord_mno(3):oEnd);
                    end
                else
                    error('dimension of the image not supported');
                end
                
                refVar = std(double(tmpImg(:)),1)^2;
                [mT,nT,oT] = size(tmpImg);
                nbPixel = mT*nT*oT;
                
                if(refVar > 0 && nbPixel > (BLOCK_WIDTH^dim)/2)
                    noiseOK = 1;
                    while noiseOK
                        %BLOCK_WIDTH_MINUS1 = in order to keep an overlap of at
                        %least 1 pixel
                        moveOK = 1;
                        
                        while moveOK
                            moveVect=round(-BLOCK_WIDTH_MINUS1 + (BLOCK_WIDTH_MINUS1+BLOCK_WIDTH_MINUS1)*rand(1,3));
                            if dim==2
                                moveVect(end)=0;
                            end
                            newBlockCoord_mno = blockCoord_mno+moveVect;
                            
                            newmStart = newBlockCoord_mno(1);
                            newnStart = newBlockCoord_mno(2);
                            newoStart = newBlockCoord_mno(3);
                            
                            newmEnd = min(m,newBlockCoord_mno(1)+BLOCK_WIDTH_MINUS1);
                            newnEnd = min(n,newBlockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                            newoEnd = min(o,newBlockCoord_mno(3)+BLOCK_WIDTH_MINUS1);
                            
                            if (newmStart >=1 && newmStart <= m...
                                    && newnStart >=1 && newnStart <= n...
                                    && newoStart >=1 && newoStart <= o)
                                [mT,nT,oT] = size(refImgImg(newmStart:newmEnd,newnStart:newnEnd,newoStart:newoEnd));
                                nbPixel = mT*nT*oT;
                                
                                if(nbPixel > (BLOCK_WIDTH^dim)/2)
                                    moveOK = 0;
                                end
                            end
                            
                        end
                        
                        newBlockCoord_xyz = [newBlockCoord_mno(1)-1,newBlockCoord_mno(2)-1, newBlockCoord_mno(3)-1];
                        %
                        referencePosition=HMatrix*[blockCoord_xyz,1]';
                        warpedPosition=HMatrix*[newBlockCoord_xyz,1]';
                        %
                        expectedBlockMatching(blockIndex,:)=[referencePosition(1) referencePosition(2) referencePosition(3) warpedPosition(1) warpedPosition(2) warpedPosition(3)];
                        expectedBlockMatchingPixel(blockIndex,:)=[blockCoord_xyz(1) blockCoord_xyz(2) blockCoord_xyz(3) newBlockCoord_xyz(1) newBlockCoord_xyz(2) newBlockCoord_xyz(3)];
                        
                        mStart=blockCoord_mno(1);
                        nStart=blockCoord_mno(2);
                        oStart=blockCoord_mno(3);
                        mEnd = min(m,blockCoord_mno(1)+BLOCK_WIDTH_MINUS1);
                        nEnd = min(n,blockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                        oEnd = min(o,blockCoord_mno(3)+BLOCK_WIDTH_MINUS1);
                        
                        newmStart=newBlockCoord_mno(1);
                        newnStart=newBlockCoord_mno(2);
                        newoStart=newBlockCoord_mno(3);
                        if dim==2
                            newoStart=1;
                        end
                        newmEnd = min(m,newBlockCoord_mno(1)+BLOCK_WIDTH_MINUS1);
                        newnEnd = min(n,newBlockCoord_mno(2)+BLOCK_WIDTH_MINUS1);
                        newoEnd = min(o,newBlockCoord_mno(3)+BLOCK_WIDTH_MINUS1);
                        if dim==2
                            newoEnd=1;
                        end
                        
                        %% Lets add noise to our floatingImage in order to not have the same block more than once
                        floImgImg(mStart:mEnd,nStart:nEnd,oStart:oEnd)=...
                            255*rand(length(mStart:mEnd),length(nStart:nEnd),length(oStart:oEnd));
                        %% Lets move the block
                        floImgImg(newmStart:newmEnd,newnStart:newnEnd,newoStart:newoEnd)=...
                            refImgImg(mStart:mStart+newmEnd-newmStart,nStart:nStart+newnEnd-newnStart,oStart:oStart+newoEnd-newoStart);
                        
                        %% Let's check that the new block does not have multiple candidate - otherwise redo
                        %% It should be 1 !
                        tmpImg1 = floImgImg(newmStart:newmEnd,newnStart:newnEnd,newoStart:newoEnd);
                        tmpImg2 = refImgImg(mStart:mStart+newmEnd-newmStart,nStart:nStart+newnEnd-newnStart,oStart:oStart+newoEnd-newoStart);
                        bestCC = corr(double(tmpImg1(:)),double(tmpImg2(:)));
                        
                        if dim == 2
                            %%
                            for vv=-3:1:3
                                for uu=-3:1:3
                                    %% we should never have edge problems
                                    currentmStart=mStart+uu;
                                    currentnStart=nStart+vv;
                                    currentmEnd=min(m,currentmStart+BLOCK_WIDTH_MINUS1);
                                    currentnEnd=min(n,currentnStart+BLOCK_WIDTH_MINUS1);
                                    
                                    if (currentmStart >=1 && currentmStart <= m && currentnStart >=1 && currentnStart <= n)
                                        tmpImg1 = floImgImg(currentmStart:currentmEnd,currentnStart:currentnEnd);
                                        tmpImg2 = refImgImg(mStart:mStart+currentmEnd-currentmStart,nStart:nStart+currentnEnd-currentnStart);
                                        currentCC = abs(corr(double(tmpImg1(:)),double(tmpImg2(:))));
                                        
                                        if (currentCC >= bestCC && currentmStart ~= newmStart && currentnStart ~= newnStart)
                                            %% redo the noise
                                            noiseOK = -1;
                                        end
                                    else
                                        disp('edge problems 2D... strange')
                                    end
                                end
                            end
                            if noiseOK==-1
                                noiseOK = 1;
                            else
                                noiseOK = 0;
                            end
                        else
                            for ww=-3:1:3
                                for vv=-3:1:3
                                    for uu=-3:1:3
                                        currentmStart=mStart+uu;
                                        currentnStart=nStart+vv;
                                        currentoStart=oStart+ww;
                                        currentmEnd=min(m,currentmStart+BLOCK_WIDTH_MINUS1);
                                        currentnEnd=min(n,currentnStart+BLOCK_WIDTH_MINUS1);
                                        currentoEnd=min(o,currentoStart+BLOCK_WIDTH_MINUS1);
                                        
                                        if (currentmStart >=1 && currentmStart <= m && currentnStart >=1 && currentnStart <= n && currentoStart >=1 && currentoStart <= o)
                                            tmpImg1 = floImgImg(currentmStart:currentmEnd,currentnStart:currentnEnd,currentoStart:currentoEnd);
                                            tmpImg2 = refImgImg(mStart:mStart+currentmEnd-currentmStart,nStart:nStart+currentnEnd-currentnStart,oStart:oStart+currentoEnd-currentoStart);
                                            currentCC = abs(corr(double(tmpImg1(:)),double(tmpImg2(:))));
                                            if (currentCC >= bestCC && currentmStart ~= newmStart && currentnStart ~= newnStart && currentoEnd ~= newoStart)
                                                %% redo the noise
                                                noiseOK = -1;
                                            end
                                        else
                                            disp('edge problems 3D... strange');
                                        end
                                    end
                                end
                            end
                            if noiseOK==-1
                                noiseOK = 1;
                            else
                                noiseOK = 0;
                            end
                        end
                        %%
                    end
                    blockIndex = blockIndex+1;
                end
            end
        end
    end
    %% SAVE THE EXPECTED BLOCK MATCHING MATRIX
    dlmwrite(strcat([output_path, '/expectedBlockMatching_mat',num2str(dim),'D.txt']), ...
        expectedBlockMatching,'precision','%.6f','delimiter',' ');
    %% SAVE THE IMAGE
    floImg.img = floImgImg;
    save_untouch_nii(floImg, strcat([output_path, '/warpedBlockMatchingImg',num2str(dim),'D.nii.gz']));
    
    indexImage=indexImage+1;
end