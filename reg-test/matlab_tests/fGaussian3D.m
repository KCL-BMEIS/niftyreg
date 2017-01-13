function h = fGaussian3D(siz, sigma)
%
if(nargin == 1)
    if length(siz) ~= 1
        error('size not well defined');
    end
    sig = siz/(4*sqrt(2*log(2)));
elseif nargin == 2
    if length(siz) ~= 1
        error('size not well defined');
    end
    if length(sigma) == 1
        sig = sigma;
    else
        error('sigma not well defined');
    end
else
    error('not enought input arguments');
end
%
size = (siz-1)/2 * ones(1,3);
[x,y,z] = ndgrid(-size(1):size(1),-size(2):size(2),-size(3):size(3));
arg = -(x.*x + y.*y + z.*z)/(2*sig*sig);
h = exp(arg);
h(h<eps*max(h(:))) = 0;
%
sumh = sum(h(:));
if sumh ~= 0
    h  = h/sumh;
end
%
end
