function LTS_test(output_path)
%% Let's generate two sets of points
%% 2-D and then 3D
nPoints = 100;
max_iter = 30;
%
for dimData = [2 3]
    %-5 + (5+5)*rand(10,1)
    P1=0 + 10*rand(nPoints,dimData);
    P1=single(P1);
    P2=0 + 10*rand(nPoints,dimData);
    P2=single(P2);
    %%
    dlmwrite(strcat([output_path,'/P1_',num2str(dimData),'D.txt']),P1,'precision','%.6f','delimiter',' ');
    dlmwrite(strcat([output_path,'/P2_',num2str(dimData),'D.txt']),P2,'precision','%.6f','delimiter',' ');
    %% READ THE FILE THAT WE HAVE JUST WRITTEN
    fileID = fopen(strcat([output_path,'/P1_',num2str(dimData),'D.txt']),'r');
    formatSpec = '%f';
    size = [dimData nPoints];
    P1 = fscanf(fileID,formatSpec,size);
    P1 = single(P1');
    fclose(fileID);
    fileID = fopen(strcat([output_path,'/P2_',num2str(dimData),'D.txt']),'r');
    formatSpec = '%f';
    size = [dimData nPoints];
    P2 = fscanf(fileID,formatSpec,size);
    P2 = single(P2');
    fclose(fileID);
    for percent_to_keep = [100 70]
        %%
        MR = LTSrigid_affine(P1, P2, percent_to_keep, max_iter, 0);
        %% SAVE THE MATRIX
        dlmwrite(strcat([output_path,'/expectedRigidLTS_',num2str(dimData),'D_',num2str(percent_to_keep),'.txt']),MR,'precision','%.6f','delimiter',' ');
        %%
        MA = LTSrigid_affine(P1, P2, percent_to_keep, max_iter, 1);
        %% SAVE THE MATRIX
        dlmwrite(strcat([output_path,'/expectedAffineLTS_',num2str(dimData),'D_',num2str(percent_to_keep),'.txt']),MA,'precision','%.6f','delimiter',' ');
        %%
    end
end