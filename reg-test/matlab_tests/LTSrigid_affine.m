function M = LTSrigid_affine(p1, p2, percent_to_keep, max_iter, isAffine)
%%
%p1=[x1,y1;
%    x1',y1';
%   ...];
%p2=[x2,y2;
%    x2',y2'];
%
%% 2D or 3D
dim=size(p1,2);
if dim>3
    error('we only handle 2D or 3D points');
end
%
nTotalPoints = size(p1,1);
nConsideredPoints = floor(percent_to_keep*nTotalPoints/100);
%Initialisation
%We extract a sublist of size nConsideredPoints from the original list, for
%example the first nConsideredPoints and we estimate the transformation:
p1Temp = p1(1:nTotalPoints,:);
p2Temp = p2(1:nTotalPoints,:);
if(isAffine)
    M1 = LSaffine(p1Temp,p2Temp);
else
    M1 = LSrigid(p1Temp,p2Temp);
end
M2 = M1;
%
compteur = 0;
lastDistance = single(Inf);
tol=single(0.001);
%
while compteur < max_iter
    %We calculate and sort the residuals for all the points:
    %r=abs(p2-(Rp1+T))
    if dim == 2
        residalArray=single([double(p2');zeros(1,nTotalPoints);ones(1,nTotalPoints)]-double(M2)*[double(p1');zeros(1,nTotalPoints);ones(1,nTotalPoints)]);
        residalArray=double(residalArray);
        residalArray=single((residalArray(1,:).^2+residalArray(2,:).^2).^(0.5));
    else
        residalArray=single([double(p2');ones(1,nTotalPoints)]-double(M2)*[double(p1');ones(1,nTotalPoints)]);
        residalArray=double(residalArray);
        residalArray=single((residalArray(1,:).^2+residalArray(2,:).^2+residalArray(3,:).^2).^(0.5));
    end
    %residalArray=(residalArray(1,:).^2+residalArray(2,:).^2);
    [residalArraySorted,id_sort] = sort(double(residalArray));
    residalArraySorted=single(residalArraySorted);
    id_sort=single(id_sort);
    distance=single(sum(double(residalArraySorted(1:nConsideredPoints))));
    if((distance > lastDistance) || (lastDistance - distance) < tol)
        M2=M1;
        break;
    end
    lastDistance = distance;
    M1=M2;
    %We select the first nConsideredPoints with the smalest residuals
    p1Temp = p1(id_sort(1:nConsideredPoints),:);
    p2Temp = p2(id_sort(1:nConsideredPoints),:);
    %We recompute the transformation
    if(isAffine)
        M2 = LSaffine(p1Temp,p2Temp);
    else
        M2 = LSrigid(p1Temp,p2Temp);
    end
    %
    compteur = compteur+1;
end
%
M=M2;
%
end