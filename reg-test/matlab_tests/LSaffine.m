function M = LSaffine(p1,p2)
%p1=[x1,y1;
%    x1',y1';
%   ...];
%p2=[x2,y2;
%    x2',y2'];
%%
%p2 = Affine*p1;
%%
%% 2D or 3D
dim=size(p1,2);
numPoints = size(p1,1);
numEquations = numPoints*dim;
if(dim==2)
    %% CREATE THE MATRIX A
    A=zeros(numEquations,6,'single');
    compteur = 1;
    for ii=1:numPoints
        A(compteur,:)=[p1(ii,:) 1 0 0 0];
        A(compteur+1,:)=[0 0 0 p1(ii,:) 1];
        compteur=compteur+2;
    end
    %
elseif(dim==3)
    %% CREATE THE MATRIX A
    A=zeros(numEquations,12,'single');
    compteur = 1;
    for ii=1:numPoints
        A(compteur,:)  =[p1(ii,:) 1 0 0 0 0 0 0 0 0];
        A(compteur+1,:)=[0 0 0 0 p1(ii,:) 1 0 0 0 0];
        A(compteur+2,:)=[0 0 0 0 0 0 0 0 p1(ii,:) 1];
        compteur=compteur+3;
    end
else
    error('LSaffine - dim > 3 not supported')
end
%%
%% Pseudo inverse:
%% A+ = V*inv(Sig)*U';
%%
%SVD de A : A=U*Sig*V'
[U,Sig,V] = svd(double(A),'econ');
U=single(U);
Sig=single(Sig);
V=single(V);
Sig(Sig(:)< 0.0001)=0;
%% 1st inv
Sig=single(inv(double(Sig)));
%% mul
tmpMulMat = single(double(V)*double(Sig));
%% last mul
Aplus = single(double(tmpMulMat)*double(U)');
%
B=single(double(p2)');
B=single(double(B(:)));
%
S = single(double(Aplus)*double(B));
%%
%Transform matrix
M = eye(4,4,'single');
if dim==2
    M(1,1)= S(1);
    M(1,2)= S(2);
    M(1,4)= S(3);
    M(2,1)= S(4);
    M(2,2)= S(5);
    M(2,4)= S(6);
    %M(3,3)=0;
else
    M(1,1)= S(1);
    M(1,2)= S(2);
    M(1,3)= S(3);
    M(1,4)= S(4);
    M(2,1)= S(5);
    M(2,2)= S(6);
    M(2,3)= S(7);
    M(2,4)= S(8);
    M(3,1)= S(9);
    M(3,2)= S(10);
    M(3,3)= S(11);
    M(3,4)= S(12);
end
end