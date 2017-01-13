function getBestLabelling_test(output_path)
%% 3 nodes - 5 displacements
nbNodes = 4;
vectDisplacement = -2:1:2;
nbDisplacement = length(vectDisplacement);
nbDisplacement3D = nbDisplacement*nbDisplacement*nbDisplacement;
vectDisplacementsArray = combvec(vectDisplacement,vectDisplacement,vectDisplacement,vectDisplacement);
% x
% |
% x--x
% |
% x
% indexToLabelArray
id_matLabel = 1;
matLabel = [];
for ii=1:nbDisplacement3D
    for jj=1:nbDisplacement3D
        for kk=1:nbDisplacement3D
            for ll=1:nbDisplacement3D
                matLabel(:,id_matLabel) = [ii-1;jj-1;kk-1;ll-1];
                %matLabel(:,id_matLabel) = [ii-1;jj-1;kk-1];
                id_matLabel = id_matLabel + 1;
            end
        end
    end
end
%% 3 nodes with random dataCost
dataCostArray = 100*rand(nbNodes,nbDisplacement3D);
dataCostArrayLinear = [];
for ii=1:size(dataCostArray,1)
    %dataCostArrayLinear = [dataCostArrayLinear,dataCostArray(end-ii+1,:)];
    dataCostArrayLinear = [dataCostArrayLinear,dataCostArray(ii,:)];
end
%% for each node and for each displacement
costFun = [];
index_costFun = 1;
for ii=1:nbDisplacement3D
    dataCost1 = dataCostArray(1,ii);
    for jj=1:nbDisplacement3D
        dataCost2 = dataCostArray(2,jj);
        for kk=1:nbDisplacement3D
            dataCost3 = dataCostArray(3,kk);
            for ll=1:nbDisplacement3D
                dataCost4 = dataCostArray(4,ll);
                costFun(index_costFun) = dataCost1 + dataCost2 + dataCost3 + dataCost4 + ...
                                         norm(vectDisplacementsArray(:,ii)'-vectDisplacementsArray(:,jj)')^2 + ...
                                         norm(vectDisplacementsArray(:,jj)'-vectDisplacementsArray(:,kk)')^2 + ...
                                         norm(vectDisplacementsArray(:,jj)'-vectDisplacementsArray(:,ll)')^2;
                %costFun(index_costFun) = dataCost1 + dataCost2 + dataCost3 + ...
                %                         norm(vectDisplacementsArray(:,ii)'-vectDisplacementsArray(:,jj)')^2 + ...
                %                         norm(vectDisplacementsArray(:,jj)'-vectDisplacementsArray(:,kk)')^2;
                index_costFun = index_costFun + 1;
            end
        end
    end
end
%
%% FIND THE MINIMUM
[minCostFun,IndexCostFun] = min(costFun);
%% THE BEST LABELING IS !!!
% index to labeling
%% BEST LABELING =
expectedLabeling=matLabel(:,IndexCostFun);
%% SAVE THE RESULTS
fileID = fopen([output_path,'/dataCost.dat'],'w');
fwrite(fileID,dataCostArrayLinear,'single');
fclose(fileID);
fileID = fopen([output_path,'/expectedLabeling.dat'],'w');
fwrite(fileID,expectedLabeling,'int');
fclose(fileID);
%%
