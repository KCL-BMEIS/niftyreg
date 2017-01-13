function svd_test(output_path)
%%
%% SVD decomposition
%%
maxSizeMatrix=10;%arbitrary value
m=randi(maxSizeMatrix);
n=randi(maxSizeMatrix);
%
A=single(double(rand(m,n,'single'))+ double(eye(m,n,'single')));
%
%% Save to file now
dlmwrite([output_path,'/inputSVDMatrix.txt'],A,'precision','%.6f','delimiter',' ');
%% READ THE FILE THAT WE HAVE JUST WRITTEN
fileID = fopen([output_path,'/inputSVDMatrix.txt'],'r');
formatSpec = '%f';
sizeA = [n m];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
A=single(A');
[U,S,V] = svd(double(A),'econ');
U=single(U);
S=single(S);
V=single(V);
%% DEBUG
%A,U,S,V
%% DEBUG
%% Save to file now
dlmwrite([output_path,'/inputSVDMatrix.txt'],A,'precision','%.6f','delimiter',' ');
dlmwrite([output_path,'/expectedUMatrix.txt'],U,'precision','%.6f','delimiter',' ');
dlmwrite([output_path,'/expectedSMatrix.txt'],S,'precision','%.6f','delimiter',' ');
dlmwrite([output_path,'/expectedVMatrix.txt'],V,'precision','%.6f','delimiter',' ');