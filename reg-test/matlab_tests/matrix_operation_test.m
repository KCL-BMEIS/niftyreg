function matrix_operation_test(output_path)
%%
%% mat44 operation tests
%%
m=4;
M1 = rand(m,m,'single')+ eye(4,4,'single');
M1(end,:)=[0 0 0 1];
M2 = rand(m,m,'single')+ eye(4,4,'single');
dlmwrite([output_path,'/inputMatrix1.txt'],M1,'precision','%.6f','delimiter',' ');
dlmwrite([output_path,'/inputMatrix2.txt'],M2,'precision','%.6f','delimiter',' ');
%% READ THE FILES THAT WE HAVE JUST WRITTEN
fileID = fopen([output_path,'/inputMatrix1.txt'],'r');
formatSpec = '%f';
sizeM1 = [m m];
M1 = fscanf(fileID,formatSpec,sizeM1);
fclose(fileID);
M1=single(M1');
fileID2 = fopen([output_path,'/inputMatrix2.txt'],'r');
formatSpec = '%f';
sizeM2 = [m m];
M2 = fscanf(fileID2,formatSpec,sizeM2);
fclose(fileID2);
M2=single(M2');
%% 1. Multiplication
expectedMul = single(double(M1)*double(M2));
dlmwrite([output_path,'/expectedMulMatrix.txt'],expectedMul,'precision','%.6f','delimiter',' ');
%% 2. Addition
expectedAdd = single(double(M1)+double(M2));
dlmwrite([output_path,'/expectedAddMatrix.txt'],expectedAdd,'precision','%.6f','delimiter',' ');
%% 3. Subtraction
expectedSub = single(double(M1)-double(M2));
dlmwrite([output_path,'/expectedSubMatrix.txt'],expectedSub,'precision','%.6f','delimiter',' ');
%% 4. Exp
expectedExp = single(expm(double(M1)));
dlmwrite([output_path,'/expectedExpMatrix.txt'],expectedExp,'precision','%.6f','delimiter',' ');
%% 5. Log
expectedLog = single(logm(double(M1)));
dlmwrite([output_path,'/expectedLogMatrix.txt'],expectedLog,'precision','%.6f','delimiter',' ');
%% 6. Inv
expectedInv = single(inv(double(M1)));
dlmwrite([output_path,'/expectedInvMatrix.txt'],expectedInv,'precision','%.6f','delimiter',' ');
