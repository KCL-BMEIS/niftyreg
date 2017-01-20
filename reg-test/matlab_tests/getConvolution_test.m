function getConvolution_test(refImg2D_name, refImg3D_name, output_path)
%% Apply convolution to the input images
%% Read the input images
input_image_name = {refImg2D_name, refImg3D_name};
%% Initialse the kernels
convolution_type = {'mea', 'lin', 'gau', 'spl'};
%% Mean kernel computation
med_kernel=ones(11,11,11) ./ 125;
%% Linear kernel computation
lin_function = [0:0.2:1, 0.8:-0.2:0];
lin_kernel=zeros(length(lin_function), ...
    length(lin_function), ...
    length(lin_function));
for z=1:length(lin_function)
    for y=1:length(lin_function)
        for x=1:length(lin_function)
            lin_kernel(x, y, z) = lin_function(x) * ...
                lin_function(y) * lin_function(z);
        end
    end
end
%% Gaussian kernel computation
gau_kernel=zeros(31,31,31);
gauss_function = gaussmf(-15:1:15, [5, 0]);
gauss_function = gauss_function ./ sum(gauss_function);
for z=1:31
    for y=1:31
        for x=1:31
            gau_kernel(x,y,z) = gauss_function(x) * ...
                gauss_function(y) * ...
                gauss_function(z);
        end
    end
end
%% Spline kernel
spline_function = zeros(length(-2:0.2:2),1);
for x=-10:1:10
    dist=abs(x/5);
    if dist < 1
         spline_function(x+11) = 2/3 - dist*dist + 0.5*dist*dist*dist;
    elseif dist < 2
        spline_function(x+11) = -(dist-2)*(dist-2)*(dist-2)/6;
    end
end
spline_function = spline_function / sum(spline_function);
spl_kernel=zeros(length(spline_function), ...
    length(spline_function), ...
    length(spline_function));
for z=1:length(spline_function)
    for y=1:length(spline_function)
        for x=1:length(spline_function)
            spl_kernel(x, y, z) = spline_function(x) * ...
                spline_function(y) * spline_function(z);
        end
    end
end
%% Loop over dimension
convolution_kernel ={med_kernel, lin_kernel, gau_kernel, spl_kernel};
for i=1:2
    %% Load the input data
    input_image = load_untouch_nii(input_image_name{i});
    input_data = input_image.img;
    %% Loop over the convolution type
    for c=1:4
        output_data = convn(input_data, convolution_kernel{c}, 'same');
        output_norm = convn(ones(size(input_data)), convolution_kernel{c}, 'same');
        output_data = output_data ./ output_norm;
        input_matrix(1,:)=input_image.hdr.hist.srow_x;
        input_matrix(2,:)=input_image.hdr.hist.srow_y;
        input_matrix(3,:)=input_image.hdr.hist.srow_z;
        convolved_nii=make_nii(output_data,...
            [input_image.hdr.dime.pixdim(2),...
            input_image.hdr.dime.pixdim(3),...
            input_image.hdr.dime.pixdim(4)],...
            [], ...
            16 ...
            );
        convolved_nii.hdr.dime.pixdim(1)=input_image.hdr.dime.pixdim(1);
        convolved_nii.hdr.hist.quatern_b=input_image.hdr.hist.quatern_b;
        convolved_nii.hdr.hist.quatern_c=input_image.hdr.hist.quatern_c;
        convolved_nii.hdr.hist.quatern_d=input_image.hdr.hist.quatern_d;
        convolved_nii.hdr.hist.qoffset_x=input_image.hdr.hist.qoffset_x;
        convolved_nii.hdr.hist.qoffset_y=input_image.hdr.hist.qoffset_y;
        convolved_nii.hdr.hist.qoffset_z=input_image.hdr.hist.qoffset_z;
        convolved_nii.hdr.hist.srow_x=input_matrix(1,:);
        convolved_nii.hdr.hist.srow_y=input_matrix(2,:);
        convolved_nii.hdr.hist.srow_z=input_matrix(3,:);
        convolved_nii.hdr.hist=input_image.hdr.hist;
        save_nii(convolved_nii, [output_path,'/convolution', ...
            int2str(i+1), 'D_', convolution_type{c}, '.nii.gz']);
    end
end
