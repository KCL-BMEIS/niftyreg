function M = LSrigid(p1,p2)
%p1=[x1,y1;
%    x1',y1';
%   ...];
%p2=[x2,y2;
%    x2',y2'];
%%
%p2 = Rp1+T;
%% 2D or 3D
dim=size(p1,2);
%%
mean_p1=single(mean(double(p1)));
mean_p2=single(mean(double(p2)));
%Coord bary:
if dim == 2
    coordBary_p1=single([double(p1(:,1))-double(mean_p1(1)),double(p1(:,2))-double(mean_p1(2))]);
    coordBary_p2=single([double(p2(:,1))-double(mean_p2(1)),double(p2(:,2))-double(mean_p2(2))]);
elseif dim ==3
    coordBary_p1=single([double(p1(:,1))-double(mean_p1(1)), double(p1(:,2))-double(mean_p1(2)), double(p1(:,3))-double(mean_p1(3))]);
    coordBary_p2=single([double(p2(:,1))-double(mean_p2(1)), double(p2(:,2))-double(mean_p2(2)), double(p2(:,3))-double(mean_p2(3))]);
else
    error('we only handle 2D or 3D points');
end
%Covariance matrix (2,2) = (2,n)*(n,2);
S=single(double(coordBary_p1)'*double(coordBary_p2));
%SVD de S : S=U*Sig*V'
[U,Sig,V] = svd(double(S),'econ');
U=single(U);
Sig=single(Sig);
V=single(V);
R=single(double(V)*double(U)');
if(det(double(R))<0)
    %disp('reflection case');
    V(:,end)=single(-double(V(:,end)));
    R=single(double(V)*double(U)');
end
%
T=single(double(mean_p2)'-double(R)*double(mean_p1)');
%% Verification
%figure;
%hold on;
%plot(p2(:,1),p2(:,2),'b+');
%p1Transform = R*p1'+repmat(T,1,size(p1',2));
%p1Transform = p1Transform';
%plot(p1Transform(:,1),p1Transform(:,2),'r+');
%%
%Transform matrix
M = eye(4,4,'single');
if dim==2
    M(1:2,1:2)=R;
    M(1,4)= T(1);
    M(2,4)= T(2);
    %M(3,3)=0;
else
    M(1:3,1:3)=R;
    M(1,4)= T(1);
    M(2,4)= T(2);
    M(3,4)= T(3);
end
%
end