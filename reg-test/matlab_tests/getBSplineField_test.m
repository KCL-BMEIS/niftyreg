function getBSplineField_test(refImg2D_name, refImg3D_name, output_path)
input_image_name={refImg2D_name, refImg3D_name};
spacing=5;
%% Loop over the input images
for i=1:2
    %% Read the input image
    input_image = load_untouch_nii(input_image_name{i});
    %% Generate the required deformation field image    
    input_ndim=2;
    if input_image.hdr.dime.dim(4) > 1
        input_ndim=3;
    end
    input_dim=[input_image.hdr.dime.dim(2), ...
        input_image.hdr.dime.dim(3), ...
        input_image.hdr.dime.dim(4) ...
        ];
    expectedField = zeros(input_dim(1), ...
        input_dim(2), ...
        input_dim(3),...
        1,...
        input_ndim,...
        'single' ...
        );
    %% Overlay a control point grid of random value
    grid_dim=[ceil(3+input_dim(1)/spacing), ...
        ceil(3+input_dim(2)/spacing), ...
        ceil(3+input_dim(3)/spacing), ...
        ];
    if input_ndim==2
           grid_dim(3)=1;
    end
    gridField = random('unif', -spacing, spacing, ...
        grid_dim(1), ...
        grid_dim(2), ...
        grid_dim(3),...
        1,...
        input_ndim ...
        );
    % Convert from displacement to deformation
    def_matrix = eye(4,4,'single');
    def_matrix(1,:)=input_image.hdr.hist.srow_x;
    def_matrix(2,:)=input_image.hdr.hist.srow_y;
    def_matrix(3,:)=input_image.hdr.hist.srow_z;
    spl_matrix = eye(4,4,'single');
    spl_matrix(1:3, 1) = def_matrix(1:3, 1) .* spacing;
    spl_matrix(1:3, 2) = def_matrix(1:3, 2) .* spacing;
    spl_matrix(1:3, 3) = def_matrix(1:3, 3) .* spacing;
    spl_matrix(:,4) = def_matrix(:,4);
    new_origin = spl_matrix * [-1, -1, -1, 1]';
    spl_matrix(:,4) = new_origin;
    for kk=1:grid_dim(3)
        for jj=1:grid_dim(2)
            for ii=1:grid_dim(1)
                newPosition = single(double(spl_matrix) * ...
                    double([ii-1 jj-1 kk-1 1]'));
                gridField(ii,jj,kk,1,1)= ...
                    gridField(ii,jj,kk,1,1)+newPosition(1);
                gridField(ii,jj,kk,1,2)= ...
                    gridField(ii,jj,kk,1,2)+newPosition(2);
                if grid_dim(3) > 1
                    gridField(ii,jj,kk,1,3)= ...
                        gridField(ii,jj,kk,1,3)+newPosition(3);
                end
            end
        end
    end
    %% Fill the deformation field image using the slowest approach
    % known to mankind
    for x=0:input_dim(1)-1
        first_x = floor(x/spacing);
        norm_x = x/spacing - first_x;
        basis_x = getBSplineCoefficient(norm_x);
        for y=0:input_dim(2)-1
            first_y = floor(y/spacing);
            norm_y = y/spacing - first_y;
            basis_y = getBSplineCoefficient(norm_y);
       
            if input_ndim==2
                current_value_x=0;
                current_value_y=0;
                for a=1:4
                    for b=1:4
                        basis = basis_x(a) * basis_y(b);
                        current_value_x=current_value_x + basis * ...
                            gridField(first_x+a, first_y+b, 1, 1, 1);
                        current_value_y=current_value_y + basis * ...
                            gridField(first_x+a, first_y+b, 1, 1, 2);
                    end
                end
                expectedField(x+1, y+1, 1, 1, 1)=current_value_x;
                expectedField(x+1, y+1, 1, 1, 2)=current_value_y;
            else
                for z=0:input_dim(3)-1
                    first_z = floor(z/spacing);
                    norm_z = z/spacing - first_z;
                    basis_z = getBSplineCoefficient(norm_z);
                    
                    current_value_x=0;
                    current_value_y=0;
                    current_value_z=0;
                    for a=1:4
                        for b=1:4
                            for c=1:4
                                basis = basis_x(a) * basis_y(b) * ...
                                    basis_z(c);
                                current_value_x=current_value_x+basis*...
                                    gridField(first_x+a, ...
                                    first_y+b, ...
                                    first_z+c, ...
                                    1, 1);
                                current_value_y=current_value_y+basis*...
                                    gridField(first_x+a, ...
                                    first_y+b, ...
                                    first_z+c, ...
                                    1, 2);
                                current_value_z=current_value_z+basis*...
                                    gridField(first_x+a, ...
                                    first_y+b, ...
                                    first_z+c, ...
                                    1, 3);
                            end
                        end
                    end
                    expectedField(x+1, y+1, z+1, 1, 1)=current_value_x;
                    expectedField(x+1, y+1, z+1, 1, 2)=current_value_y;
                    expectedField(x+1, y+1, z+1, 1, 3)=current_value_z;
                end
            end            
        end
    end
    %% Save the deformation field image
    expectedField_nii=make_nii(expectedField,...
        [input_image.hdr.dime.pixdim(2),...
         input_image.hdr.dime.pixdim(3),...
         input_image.hdr.dime.pixdim(4)],...
        [], ...
        16 ...
        );
    expectedField_nii.hdr.dime.pixdim(1)=input_image.hdr.dime.pixdim(1);
    expectedField_nii.hdr.hist.quatern_b=input_image.hdr.hist.quatern_b;
    expectedField_nii.hdr.hist.quatern_c=input_image.hdr.hist.quatern_c;
    expectedField_nii.hdr.hist.quatern_d=input_image.hdr.hist.quatern_d;
    expectedField_nii.hdr.hist.qoffset_x=input_image.hdr.hist.qoffset_x;
    expectedField_nii.hdr.hist.qoffset_y=input_image.hdr.hist.qoffset_y;
    expectedField_nii.hdr.hist.qoffset_z=input_image.hdr.hist.qoffset_z;
    expectedField_nii.hdr.hist.srow_x=def_matrix(1,:);
    expectedField_nii.hdr.hist.srow_y=def_matrix(2,:);
    expectedField_nii.hdr.hist.srow_z=def_matrix(3,:);
    expectedField_nii.hdr.hist=input_image.hdr.hist;
    save_nii(expectedField_nii, [output_path,'/bspline_def', ...
        int2str(input_ndim), 'D.nii.gz']);
    %% Save the control point grid
    gridField_nii=make_nii(gridField,...
        [spacing*input_image.hdr.dime.pixdim(2),...
         spacing*input_image.hdr.dime.pixdim(3),...
         spacing*input_image.hdr.dime.pixdim(4)],...
        [], ...
        16 ...
        );
    gridField_nii.hdr.dime.pixdim(1)=input_image.hdr.dime.pixdim(1);
    gridField_nii.hdr.hist.quatern_b=input_image.hdr.hist.quatern_b;
    gridField_nii.hdr.hist.quatern_c=input_image.hdr.hist.quatern_c;
    gridField_nii.hdr.hist.quatern_d=input_image.hdr.hist.quatern_d;
    gridField_nii.hdr.hist.qoffset_x=input_image.hdr.hist.qoffset_x;
    gridField_nii.hdr.hist.qoffset_y=input_image.hdr.hist.qoffset_y;
    gridField_nii.hdr.hist.qoffset_z=input_image.hdr.hist.qoffset_z;
    gridField_nii.hdr.hist=input_image.hdr.hist;
    gridField_nii.hdr.hist.srow_x=spl_matrix(1,:);
    gridField_nii.hdr.hist.srow_y=spl_matrix(2,:);
    gridField_nii.hdr.hist.srow_z=spl_matrix(3,:);
    save_nii(gridField_nii, [output_path,'/bspline_grid', ...
        int2str(input_ndim), 'D.nii.gz']);
end

function basis = getBSplineCoefficient(dist)
%% Given a normalise position return the 4 corresponding basis values
basis(1) = (1-dist)*(1-dist)*(1-dist)/6;
basis(2) = (3*dist*dist*dist - 6*dist*dist + 4)/6.0;
basis(3) = (-3*dist*dist*dist + 3*dist*dist + 3*dist + 1)/6;
basis(4) = dist*dist*dist/6;
