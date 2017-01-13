function getMST_test(output_path)
% %%
size_x = 5;
size_y = 3;
size_z = 2;
nbNodes = 15*2;
nbEdges = nbNodes*6;
index_neighbours = -1*ones(1,nbEdges);
edgeWeightMatrix = zeros(1,nbEdges);
%%
M1=[1 1 2  2 3 3 4 4 5  6 6  7  7  8  8  9  9 10  11 12 13 14]+15;
M2=[2 6 3  7 4 8 5 9 10 7 11 8 12  9 13 10 14 15  12 13 14 15]+15;
E1 = [1 1 2  2 3 3 4 4 5  6 6  7  7  8  8  9  9 10  11 12 13 14, 1  2    3  4  5  6   7   8   9  10  11 12 13 14 15, M1];
E2 = [2 6 3  7 4 8 5 9 10 7 11 8 12  9 13 10 14 15  12 13 14 15, 16 17  18 19 20 21  22  23  24  25  26 27 28 29 30, M2];
% %% DEBUG
% size_x = 2;
% size_y = 3;
% size_z = 1;
% nbNodes = 6;
% nbEdges = nbNodes*6;
% index_neighbours = -1*ones(1,nbEdges);
% edgeWeightMatrix = zeros(1,nbEdges);
% E1 = [1 1 2 3 3 4 5];
% E2 = [2 3 4 4 5 6 6];
% %% DEBUG
A=zeros(nbNodes,nbNodes);
for ii=1:length(E1)
    A(E1(ii),E2(ii))=rand(1);
end
%
A=A+A';
%
for ii=1:nbNodes
    for jj=1:nbNodes
        if(abs(A(ii,jj)) > 0) % there is a connexion
            %% which direction ?
            diff = jj-ii;
            if(diff == -1)
                %gauche
                ngh_id = 0;
            end
            if(diff == 1)
                %droite
                ngh_id = 1;
            end
            if(diff == size_x)
                %bas
                ngh_id = 3;%2
            end
            if(diff == -size_x)
                %haut
                ngh_id = 2;%3
            end
            if(diff == -size_x*size_y)
                %-z
                ngh_id = 4;
            end
            if(diff == size_x*size_y)
                %+z
                ngh_id = 5;
            end
            index_neighbours(ii+ngh_id*nbNodes)=jj-1;
            edgeWeightMatrix(ii+ngh_id*nbNodes)=A(ii,jj);
        end
    end
end
%% SAVE THE INPUT
fileID = fopen([output_path,'/indexNeighbours.dat'],'w');
fwrite(fileID,index_neighbours,'int');
fclose(fileID);
fileID = fopen([output_path,'/EWeightMatrix.dat'],'w');
fwrite(fileID,edgeWeightMatrix,'single');
fclose(fileID);
%%
%
UG=tril(sparse(A));
%
%UG = graph(A);
%view(biograph(UG,[],'ShowArrows','off','ShowWeights','on'))
%% Let's use MATLAB function :-)
[ST, pred] = graphminspantree(UG,1);
%%
%view(biograph(ST,[],'ShowArrows','off','ShowWeights','on'))
%% SAVE THE OUTPUT
fileID = fopen([output_path,'/expectedParentsList.dat'],'w');
fwrite(fileID,pred,'int');
fclose(fileID);
