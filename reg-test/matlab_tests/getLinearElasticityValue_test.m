function getLinearElasticityValue_test(grid2D_name, ...
    def2D_name, grid3D_name, def3D_name, output_path)
%%
grid_name = {grid2D_name, grid3D_name};
defField_name = {def2D_name, def3D_name};

for i=1:2
    %Read the grid image
    grid_image = load_untouch_nii(grid_name{i});
    grid_data = grid_image.img;
    orientation = zeros(3,3);
    orientation(1:3,1) = grid_image.hdr.hist.srow_x(1:3);
    orientation(1:3,2) = grid_image.hdr.hist.srow_y(1:3);
    orientation(1:3,3) = grid_image.hdr.hist.srow_z(1:3);
    orientation = inv(orientation);
    grid_dim=[grid_image.hdr.dime.dim(2), ...
        grid_image.hdr.dime.dim(3), ...
        grid_image.hdr.dime.dim(4) ...
        ];
    
    constraint_approx = 0;
    % Precompute the basis values
    basis = getBSplineCoefficient(0);
    first = getBSplineCoefficientFirstOrder(0);
    % Compute the value at the control point position only
    for x=2:grid_dim(1)-1
        for y=2:grid_dim(2)-1
            if (i+1)==2
                jacobian = zeros(2,2);
                for a=1:3
                    for b=1:3
                        jacobian(1,1)=jacobian(1,1) + ...
                            first(a) * basis(b) * ...
                            grid_data(x+a-2, y+b-2, 1, 1, 1);
                        jacobian(1,2)=jacobian(1,2) + ...
                            basis(a) * first(b) * ...
                            grid_data(x+a-2, y+b-2, 1, 1, 1);
                        jacobian(2,1)=jacobian(2,1) + ...
                            first(a) * basis(b) * ...
                            grid_data(x+a-2, y+b-2, 1, 1, 2);
                        jacobian(2,2)=jacobian(2,2) + ...
                            basis(a) * first(b) * ...
                            grid_data(x+a-2, y+b-2, 1, 1, 2);
                    end
                end
                jacobian = orientation(1:2,1:2) * jacobian';
                rotation = polarDecomposition(jacobian);
                jacobian = (rotation) \ jacobian;
                jacobian = jacobian - eye(2);
                for a=1:2
                    for b=1:2
                        constraint_approx = constraint_approx + ...
                            (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                    end
                end
            else
                for z=2:grid_dim(3)-1
                    jacobian = zeros(3,3);
                    for a=1:3
                        for b=1:3
                            for c=1:3
                                jacobian(1,1)=jacobian(1,1) +  ...
                                    first(a) * basis(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 1);
                                jacobian(1,2)=jacobian(1,2) +  ...
                                    basis(a) * first(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 1);
                                jacobian(1,3)=jacobian(1,3) +  ...
                                    basis(a) * basis(b) * first(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 1);
                                
                                jacobian(2,1)=jacobian(2,1) +  ...
                                    first(a) * basis(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 2);
                                jacobian(2,2)=jacobian(2,2) +  ...
                                    basis(a) * first(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 2);
                                jacobian(2,3)=jacobian(2,3) +  ...
                                    basis(a) * basis(b) * first(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 2);
                                
                                jacobian(3,1)=jacobian(3,1) +  ...
                                    first(a) * basis(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 3);
                                jacobian(3,2)=jacobian(3,2) +  ...
                                    basis(a) * first(b) * basis(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 3);
                                jacobian(3,3)=jacobian(3,3) +  ...
                                    basis(a) * basis(b) * first(c) * ...
                                    grid_data(x+a-2, y+b-2, z+c-2, 1, 3);
                            end
                        end
                    end
                    jacobian = orientation * jacobian';
                    rotation = polarDecomposition(jacobian);
                    jacobian = (rotation) \ jacobian;
                    jacobian = jacobian - eye(3);
                    for a=1:3
                        for b=1:3
                            constraint_approx = constraint_approx + ...
                                (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                        end
                    end
                end
            end
        end
    end
    dlmwrite([output_path,'/le_spline_approx',num2str(i+1),'D.txt'], ...
        constraint_approx/ numel(grid_data), ...
        'precision','%.6f','delimiter',' ');
    
    %Read the deformation field image
    def_image = load_untouch_nii(defField_name{i});
    def_data = def_image.img;
    def_dim=[def_image.hdr.dime.dim(2), ...
        def_image.hdr.dime.dim(3), ...
        def_image.hdr.dime.dim(4) ...
        ];
    spacing = grid_image.hdr.dime.pixdim(2) / def_image.hdr.dime.pixdim(2);
    constraint_dense = 0;
    
    % Compute the value at all voxel position
    for x=0:def_dim(1)-1
        pre_x = floor(x/spacing);
        norm_x = x/spacing - pre_x;
        basis_x = getBSplineCoefficient(norm_x);
        first_x = getBSplineCoefficientFirstOrder(norm_x);
        for y=0:def_dim(2)-1
            pre_y = floor(y/spacing);
            norm_y = y/spacing - pre_y;
            basis_y = getBSplineCoefficient(norm_y);
            first_y = getBSplineCoefficientFirstOrder(norm_y);
            if (i+1)==2
                jacobian = zeros(2,2);
                for a=1:4
                    for b=1:4
                        jacobian(1,1)=jacobian(1,1) + ...
                            first_x(a) * basis_y(b) * ...
                            grid_data(pre_x+a, pre_y+b, 1, 1, 1);
                        jacobian(1,2)=jacobian(1,2) + ...
                            basis_x(a) * first_y(b) * ...
                            grid_data(pre_x+a, pre_y+b, 1, 1, 1);
                        jacobian(2,1)=jacobian(2,1) + ...
                            first_x(a) * basis_y(b) * ...
                            grid_data(pre_x+a, pre_y+b, 1, 1, 2);
                        jacobian(2,2)=jacobian(2,2) + ...
                            basis_x(a) * first_y(b) * ...
                            grid_data(pre_x+a, pre_y+b, 1, 1, 2);
                    end
                end
                jacobian = orientation(1:2,1:2) * jacobian';
                rotation = polarDecomposition(jacobian);
                jacobian = (rotation) \ jacobian;
                jacobian = jacobian - eye(2);
                for a=1:2
                    for b=1:2
                        constraint_dense = constraint_dense + ...
                            (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                    end
                end
            else
                for z=0:def_dim(3)-1
                    pre_z = floor(z/spacing);
                    norm_z = z/spacing - pre_z;
                    basis_z = getBSplineCoefficient(norm_z);
                    first_z = getBSplineCoefficientFirstOrder(norm_z);
                    jacobian = zeros(3,3);
                    for a=1:4
                        for b=1:4
                            for c=1:4
                                jacobian(1,1)=jacobian(1,1) +  ...
                                    first_x(a) * basis_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 1);
                                jacobian(1,2)=jacobian(1,2) +  ...
                                    basis_x(a) * first_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 1);
                                jacobian(1,3)=jacobian(1,3) +  ...
                                    basis_x(a) * basis_y(b) * first_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 1);
                                
                                jacobian(2,1)=jacobian(2,1) +  ...
                                    first_x(a) * basis_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 2);
                                jacobian(2,2)=jacobian(2,2) +  ...
                                    basis_x(a) * first_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 2);
                                jacobian(2,3)=jacobian(2,3) +  ...
                                    basis_x(a) * basis_y(b) * first_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 2);
                                
                                jacobian(3,1)=jacobian(3,1) +  ...
                                    first_x(a) * basis_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 3);
                                jacobian(3,2)=jacobian(3,2) +  ...
                                    basis_x(a) * first_y(b) * basis_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 3);
                                jacobian(3,3)=jacobian(3,3) +  ...
                                    basis_x(a) * basis_y(b) * first_z(c) * ...
                                    grid_data(pre_x+a, pre_y+b, pre_z+c, 1, 3);
                            end
                        end
                    end
                    jacobian = orientation * jacobian';
                    rotation = polarDecomposition(jacobian);
                    jacobian = (rotation) \ jacobian;
                    jacobian = jacobian - eye(3);
                    for a=1:3
                        for b=1:3
                            constraint_dense = constraint_dense + ...
                                (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                        end
                    end
                end
            end
        end
    end
    dlmwrite([output_path,'/le_spline_dense',num2str(i+1),'D.txt'], ...
        constraint_dense/ numel(def_data), ...
        'precision','%.6f','delimiter',' ');
    clear grid_image;
    
    orientation(1:3,1) = def_image.hdr.hist.srow_x(1:3);
    orientation(1:3,2) = def_image.hdr.hist.srow_y(1:3);
    orientation(1:3,3) = def_image.hdr.hist.srow_z(1:3);
    orientation = inv(orientation);
    basis=[1,0];
    first=[-1,1];
    constraint_dense=0;
    for x=1:def_dim(1)
        if x==def_dim(1)
            X=x-1;
        else
            X=x;
        end
        for y=1:def_dim(2)
            if y==def_dim(2)
                Y=y-1;
            else
                Y=y;
            end
            if (i+1)==2
                jacobian = zeros(2,2);
                for a=1:2
                    for b=1:2
                        jacobian(1,1)=jacobian(1,1) + ...
                            first(a) * basis(b) * ...
                            def_data(X+a-1, Y+b-1, 1, 1, 1);
                        jacobian(1,2)=jacobian(1,2) + ...
                            basis(a) * first(b) * ...
                            def_data(X+a-1, Y+b-1, 1, 1, 1);
                        jacobian(2,1)=jacobian(2,1) + ...
                            first(a) * basis(b) * ...
                            def_data(X+a-1, Y+b-1, 1, 1, 2);
                        jacobian(2,2)=jacobian(2,2) + ...
                            basis(a) * first(b) * ...
                            def_data(X+a-1, Y+b-1, 1, 1, 2);
                    end
                end
                jacobian = orientation(1:2,1:2) * jacobian';
                rotation = polarDecomposition(jacobian);
                jacobian = (rotation) \ jacobian;
                jacobian = jacobian - eye(2);
                for a=1:2
                    for b=1:2
                        constraint_dense = constraint_dense + ...
                            (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                    end
                end
            else
                for z=1:def_dim(3)
                    if z==def_dim(3)
                        Z=z-1;
                    else
                        Z=z;
                    end
                    jacobian = zeros(3,3);
                    for a=1:2
                        for b=1:2
                            for c=1:2
                                jacobian(1,1)=jacobian(1,1) + ...
                                    first(a) * basis(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 1);
                                jacobian(1,2)=jacobian(1,2) + ...
                                    basis(a) * first(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 1);
                                jacobian(1,3)=jacobian(1,3) + ...
                                    basis(a) * basis(b) * first(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 1);
                                
                                jacobian(2,1)=jacobian(2,1) + ...
                                    first(a) * basis(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 2);
                                jacobian(2,2)=jacobian(2,2) + ...
                                    basis(a) * first(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 2);
                                jacobian(2,3)=jacobian(2,3) + ...
                                    basis(a) * basis(b) * first(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 2);
                                
                                jacobian(3,1)=jacobian(3,1) + ...
                                    first(a) * basis(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 3);
                                jacobian(3,2)=jacobian(3,2) + ...
                                    basis(a) * first(b) * basis(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 3);
                                jacobian(3,3)=jacobian(3,3) + ...
                                    basis(a) * basis(b) * first(c) * ...
                                    def_data(X+a-1, Y+b-1, Z+c-1, 1, 3);
                            end
                        end
                    end
                    jacobian = orientation * jacobian';
                    rotation = polarDecomposition(jacobian);
                    jacobian = (rotation) \ jacobian;
                    jacobian = jacobian - eye(3);
                    for a=1:3
                        for b=1:3
                            constraint_dense = constraint_dense + ...
                                (0.5*(jacobian(a,b)+jacobian(b,a)))^2;
                        end
                    end
                end
            end
        end
    end
    
    dlmwrite([output_path,'/le_field_dense',num2str(i+1),'D.txt'], ...
        constraint_dense/ numel(def_data), ...
        'precision','%.6f','delimiter',' ');
end

return

function R = polarDecomposition(F)
%% Polar decomposition of a given matrix
C = F'*F;
[Q0, lambdasquare] = eig(C);
lambda = sqrt(diag((lambdasquare)));
Uinv = repmat(1./lambda',size(F,1),1).*Q0*Q0';
R = F*Uinv;

function basis = getBSplineCoefficient(dist)
%% Given a normalise position return the 4 corresponding basis values
basis(1) = (1-dist)*(1-dist)*(1-dist)/6;
basis(2) = (3*dist*dist*dist - 6*dist*dist + 4)/6.0;
basis(3) = (-3*dist*dist*dist + 3*dist*dist + 3*dist + 1)/6;
basis(4) = dist*dist*dist/6;

function first = getBSplineCoefficientFirstOrder(dist)
%% Given a normalise position return the 4 corresponding basis values
first(4)= dist * dist / 2;
first(1)= dist - 0.5 - first(4);
first(3)= 1 + first(1) - 2*first(4);
first(2)= - first(1) - first(3) - first(4);