#include "_reg_mrf.h"


/*****************************************************/
reg_mrf::reg_mrf(reg_measure *_measure,
                 nifti_image *_controlPointImage,
                 int _discrete_radius,
                 int _discrete_increment,
                 float _reg_weight)
{
    this->measure = _measure;
    this->controlPointImage = _controlPointImage;
    this->discrete_radius = _discrete_radius; //default 18
    this->discrete_increment = _discrete_increment; // default 3
    this->regularisation_weight = _reg_weight;

    this->dim = this->controlPointImage->nz > 1 ? 3 :2;
    // Allocate the discretised value result
    int discrete_value = (this->discrete_radius / this->discrete_increment ) * 2 + 1;
    int nD_discrete_value = std::pow(discrete_value,dim);
    int controlPointNumber = this->controlPointImage->nx *
                             this->controlPointImage->ny * this->controlPointImage->nz;
    this->discretised_measure = (float *)malloc(controlPointNumber*nD_discrete_value*sizeof(float));

    // Allocate the arrays to store the graph
    this->edgeWeightMatrix = (float *)calloc(controlPointNumber*dim*2,sizeof(float));
    this->index_neighbours = (float *)malloc(controlPointNumber*dim*2*sizeof(float));
    for(int i =0;i<controlPointNumber*dim*2;i++) {
        this->index_neighbours[i]=-1;
    }
    this->orderedList = (int *) malloc(controlPointNumber*sizeof(int));
    this->parentsList = (int *) malloc(controlPointNumber*sizeof(int));
    this->edgeWeight = (float *) malloc(controlPointNumber*sizeof(float));

    //regulatization - optimization
    this->regularisedCost= (float *)malloc(controlPointNumber*nD_discrete_value*sizeof(float));  //probabilistic output of regularisationSGM (not yet used)
    this->optimalDisplacement=(int *)malloc(controlPointNumber*sizeof(int)); //index of optimal displacement for every control point (output2)

    this->initialised = false;
}
/*****************************************************/
reg_mrf::~reg_mrf()
{
    if(this->discretised_measure!=NULL)
        free(this->discretised_measure);
    this->discretised_measure=NULL;

    if(this->edgeWeightMatrix!=NULL)
        free(this->edgeWeightMatrix);
    this->edgeWeightMatrix=NULL;

    if(this->index_neighbours!=NULL)
        free(this->index_neighbours);
    this->index_neighbours=NULL;

    if(this->orderedList!=NULL)
        free(this->orderedList);
    this->orderedList=NULL;

    if(this->parentsList!=NULL)
        free(this->parentsList);
    this->parentsList=NULL;

    if(this->edgeWeight!=NULL)
        free(this->edgeWeight);
    this->edgeWeight=NULL;

    if(this->regularisedCost!=NULL)
        free(this->regularisedCost);
    this->regularisedCost=NULL;

    if(this->optimalDisplacement!=NULL)
        free(this->optimalDisplacement);
    this->optimalDisplacement=NULL;
}
/*****************************************************/
void reg_mrf::Initialise()
{
    // Create the minimum spamming tree
    this->GetGraph();
    this->GetPrimsMST();
    this->initialised = true;
}
/*****************************************************/
void reg_mrf::GetDiscretisedMeasure()
{
    measure->GetDiscretisedValue(this->controlPointImage,
                                 this->discretised_measure,
                                 this->discrete_radius,
                                 this->discrete_increment);
}
/*****************************************************/
void reg_mrf::Optimise()
{
    // Run the optimisation and update the transformation
    this->GetRegularisation();
    reg_print_msg_error("Need to implement reg_mrf::Optimise()");
    reg_exit();
}
/*****************************************************/
void reg_mrf::Run()
{
    if(this->initialised==false)
        this->Initialise();
    this->GetDiscretisedMeasure();
    this->Optimise();
}
/*****************************************************/
/*****************************************************/
template <class DTYPE>
void GetGraph_core3D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask)
{
    int cpx, cpy, cpz, t, x, y, z, blockIndex, voxIndex;
    float gridVox[3], imageVox[3];
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if(controlPointGridImage->sform_code>0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_xyz;
    if(refImage->sform_code>0)
        grid_vox2mm = &refImage->sto_xyz;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    int blockSize[3]={
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
    };
    int controlPointNumber = controlPointGridImage->nx*controlPointGridImage->ny*controlPointGridImage->nz;
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    // Allocate some static memory
    float refBlockValue[voxelBlockNumber];
    float neighbourBlockValue[voxelBlockNumber];
    float SADNeighbourValue = 0;

    // Pointers to the input image
    size_t voxelNumber = (size_t)refImage->nx *
                         refImage->ny * refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *currentRefPtr = NULL;

    // Loop over all control points
    for(cpz=0; cpz<controlPointGridImage->nz; ++cpz){
        gridVox[2] = cpz;
        for(cpy=0; cpy<controlPointGridImage->ny; ++cpy){
            gridVox[1] = cpy;
            for(cpx=0; cpx<controlPointGridImage->nx; ++cpx){
                gridVox[0] = cpx;
                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                // Extract the block in the reference image
                blockIndex = 0;

                for(t=0; t<refImage->nt; ++t){
                    currentRefPtr = &refImgPtr[t*voxelNumber];

                    for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                        for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                            for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                                voxIndex = (z*refImage->ny+y)*refImage->nx+x;
                                if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz && mask[voxIndex]>-1){
                                    refBlockValue[blockIndex] = currentRefPtr[voxIndex];
                                }
                                else refBlockValue[blockIndex] = 0.f;
                                ++blockIndex;
                            } // x
                        } // y
                    } // z
                } //t
                //Let look at the neighbours now -- 6 in 3D
                //standard six-neighbourhood for grid graph
                const int nb_neighbours = 6;
                int dx[nb_neighbours]={-1,1,0,0,0,0};
                int dy[nb_neighbours]={0,0,-1,1,0,0};
                int dz[nb_neighbours]={0,0,0,0,-1,1};

                for(int ngh_index=0;ngh_index<nb_neighbours;ngh_index++) {

                    gridVox[2] = cpz+dz[ngh_index];
                    gridVox[1] = cpy+dy[ngh_index];
                    gridVox[0] = cpx+dx[ngh_index];
                    // Compute the corresponding image voxel position
                    reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                    if(imageVox[0]>-1 && imageVox[0]<refImage->nx &&
                            imageVox[1]>-1 && imageVox[1]<refImage->ny &&
                            imageVox[2]>-1 && imageVox[2]<refImage->nz) {

                        blockIndex = 0;

                        for(t=0; t<refImage->nt; ++t){
                            currentRefPtr = &refImgPtr[t*voxelNumber];

                            for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                                for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                                    for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                                        voxIndex = (z*refImage->ny+y)*refImage->nx+x;
                                        if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz && mask[voxIndex]>-1){
                                            neighbourBlockValue[blockIndex] = currentRefPtr[voxIndex];
                                        }
                                        else neighbourBlockValue[blockIndex] = 0.f;
                                        ++blockIndex;
                                    } // x
                                } // y
                            } // z
                        } //t

                        SADNeighbourValue = 0;
                        for(int sadIndex=0;sadIndex<voxelBlockNumber;sadIndex++) {
                            SADNeighbourValue += std::abs(neighbourBlockValue[sadIndex]-refBlockValue[sadIndex]);
                        }
                        //store results:
                        index_neighbours[cpx+cpy*controlPointGridImage->nx+
                                cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                                ngh_index*controlPointNumber]=
                                cpx+dx[ngh_index]+(y+dy[ngh_index])*controlPointGridImage->nx+
                                (z+dz[ngh_index])*controlPointGridImage->nx*controlPointGridImage->ny;
                        edgeWeightMatrix[cpx+cpy*controlPointGridImage->nx+
                                cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                                ngh_index*controlPointNumber]=SADNeighbourValue;
                    }

                }
            } //cpx
        } //cpy
    } //cpz
    //normalise edgeweights by stddev of image ???????


}
template <class DTYPE>
void GetGraph_core2D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     float* index_neighbours,
                     nifti_image *refImage,
                     int *mask)
{
    int cpx, cpy, t, x, y, blockIndex, voxIndex;
    float gridVox[3], imageVox[3];
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if(controlPointGridImage->sform_code>0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_xyz;
    if(refImage->sform_code>0)
        grid_vox2mm = &refImage->sto_xyz;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    int blockSize[3]={
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int) 1,
    };
    int controlPointNumber = controlPointGridImage->nx*controlPointGridImage->ny*controlPointGridImage->nz;
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    // Allocate some static memory
    float refBlockValue[voxelBlockNumber];
    float neighbourBlockValue[voxelBlockNumber];
    float SADNeighbourValue = 0;

    // Pointers to the input image
    size_t voxelNumber = (size_t)refImage->nx *
                         refImage->ny * refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *currentRefPtr = NULL;

    // Loop over all control points
    gridVox[2] = 0;
    for(cpy=0; cpy<controlPointGridImage->ny; ++cpy){
        gridVox[1] = cpy;
        for(cpx=0; cpx<controlPointGridImage->nx; ++cpx){
            gridVox[0] = cpx;
            // Compute the corresponding image voxel position
            reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
            // Extract the block in the reference image
            blockIndex = 0;

            for(t=0; t<refImage->nt; ++t){
                currentRefPtr = &refImgPtr[t*voxelNumber];

                for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                    for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                        voxIndex = y*refImage->nx+x;
                        if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && mask[voxIndex]>-1){
                            refBlockValue[blockIndex] = currentRefPtr[voxIndex];
                        }
                        else refBlockValue[blockIndex] = 0.f;
                        ++blockIndex;
                    } // x
                } // y
            } // z
        } //t
        //Let look at the neighbours now -- 4 in 2D
        //standard six-neighbourhood for grid graph
        const int nb_neighbours = 4;
        int dx[nb_neighbours]={-1,1,0,0};
        int dy[nb_neighbours]={0,0,-1,1};

        for(int ngh_index=0;ngh_index<nb_neighbours;ngh_index++) {
            gridVox[1] = cpy+dy[ngh_index];
            gridVox[0] = cpx+dx[ngh_index];
            // Compute the corresponding image voxel position
            reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
            if(imageVox[0]>-1 && imageVox[0]<refImage->nx &&
                    imageVox[1]>-1 && imageVox[1]<refImage->ny) {

                blockIndex = 0;

                for(t=0; t<refImage->nt; ++t){
                    currentRefPtr = &refImgPtr[t*voxelNumber];

                    for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                        for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                            voxIndex = y*refImage->nx+x;
                            if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && mask[voxIndex]>-1){
                                neighbourBlockValue[blockIndex] = currentRefPtr[voxIndex];
                            }
                            else neighbourBlockValue[blockIndex] = 0.f;
                            ++blockIndex;
                        } // x
                    } // y
                } // t

                SADNeighbourValue = 0;
                for(int sadIndex=0;sadIndex<voxelBlockNumber;sadIndex++) {
                    SADNeighbourValue += std::abs(neighbourBlockValue[sadIndex]-refBlockValue[sadIndex]);
                }
                //store results:
                index_neighbours[
                        cpx+cpy*controlPointGridImage->nx+
                        +ngh_index*controlPointNumber]=
                        cpx+dx[ngh_index]+(y+dy[ngh_index])*controlPointGridImage->nx;
                edgeWeightMatrix[cpx+cpy*controlPointGridImage->nx+
                        +ngh_index*controlPointNumber]=SADNeighbourValue;
            }
        } //cpx
    } //cpy
    //normalise edgeweights by stddev of image ???????
}
/* *************************************************************** */
void reg_mrf::GetGraph()
{
    if(this->measure->GetReferenceImage()->nz > 1) {
        switch(this->measure->GetReferenceImage()->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetGraph_core3D<float>
                    (this->controlPointImage,
                     this->edgeWeightMatrix,
                     this->index_neighbours,
                     this->measure->GetReferenceImage(),
                     this->measure->GetReferenceMask()
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetGraph_core3D<double>
                    (this->controlPointImage,
                     this->edgeWeightMatrix,
                     this->index_neighbours,
                     this->measure->GetReferenceImage(),
                     this->measure->GetReferenceMask()
                     );
            break;
        default:
            reg_print_fct_error("reg_mrf::GetGraph");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    } else {
        switch(this->measure->GetReferenceImage()->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetGraph_core2D<float>
                    (this->controlPointImage,
                     this->edgeWeightMatrix,
                     this->index_neighbours,
                     this->measure->GetReferenceImage(),
                     this->measure->GetReferenceMask()
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetGraph_core2D<double>
                    (this->controlPointImage,
                     this->edgeWeightMatrix,
                     this->index_neighbours,
                     this->measure->GetReferenceImage(),
                     this->measure->GetReferenceMask()
                     );
            break;
        default:
            reg_print_fct_error("reg_mrf::GetGraph");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
/*****************************************************/
//CUT THE EDGES WITH HIGH COST = INTENSITY DIFFERENCES!
/*****************************************************/
void reg_mrf::GetPrimsMST()
{
    int num_vertices = this->controlPointImage->nx *
                       this->controlPointImage->ny * this->controlPointImage->nz;
    int currentNode=0; //arbritary root node -- I think this could be changed
    //list of nodes already in MST
    bool* addedToMST=new bool[num_vertices];
    for(int i=0;i<num_vertices;i++){
        addedToMST[i]=false;
    }
    addedToMST[currentNode]=true;
    std::pair<short,int>* treeLevel=new std::pair<short,int>[num_vertices];
    treeLevel[currentNode]=std::pair<short,int>(0,currentNode);

    int num_neighbours=this->controlPointImage->nz > 1 ? 6 : 4;

    this->parentsList[currentNode]=-1; //root has no parent
    std::priority_queue<Edge> priority; //priority queue - ordered list - high --- low
    //Edge comparison - a edge is inf if weight is bigger (cf. edge struct) ==> ordered from low to high weights

    float mincost=0.0f;
    //run n-1 times so that all nodes added
    for(int i=0;i<num_vertices-1;i++){
        //add edgesss of new node to priority queue
        for(int j=0;j<num_neighbours;j++){
            int index_j=this->index_neighbours[currentNode+j*num_vertices];
            float weight=this->edgeWeightMatrix[currentNode+j*num_vertices];
            //index_neighbours is initialized at -1
            if(index_j>=0){
               Edge current_edge = {weight,currentNode,index_j};
               priority.push(current_edge);//weight - start index - end index
            }

        }
        currentNode=-1;
        while(currentNode==-1){
            Edge bestEdge=priority.top();
            priority.pop();
            //test whether endIndex of edge is already in MST
            if(addedToMST[bestEdge.startIndex] && !addedToMST[bestEdge.endIndex]){
                mincost+=-bestEdge.weight;

                //edgeWeight[bestEdge.endIndex]=-bestEdge.weight;//if normalization
                this->edgeWeight[bestEdge.endIndex]=bestEdge.weight;

                currentNode=bestEdge.endIndex;
                addedToMST[bestEdge.endIndex]=true;
                this->parentsList[bestEdge.endIndex]=bestEdge.startIndex;
                treeLevel[bestEdge.endIndex]=std::pair<short,int>(treeLevel[bestEdge.startIndex].first+1,bestEdge.endIndex);

            }

        }

    }
    //generate list of nodes ordered by tree depth
    std::sort(treeLevel,treeLevel+num_vertices);
    //printf("max tree depth: %d, mincost: %f\n",treeLevel[num_vertices-1].first,mincost);
    for(int i=0;i<num_vertices;i++){
        orderedList[i]=treeLevel[i].second;
    }
    //Free memory
    free(treeLevel);
    free(addedToMST);
}
/*****************************************************/
void reg_mrf::GetRegularisation()
{
    /* Incremental diffusion regularisation of parametrised transformation
     using (globally optimal) belief-propagation on minimum spanning tree.
     Fast distance transform (see fastDT2.h) uses squared differences.
     Similarity cost for each node and label has to be given as input.
    */
    //dense displacement space
    int label_len=(this->discrete_radius / this->discrete_increment ) * 2 + 1; //length and total size of displacement space
    int label_num=std::pow(label_len,this->dim);

    //buffer variable
    float *cost1=new float[label_num];
    float *vals=new float[label_num];
    int *inds=new int[label_num];

    //array of messages
    int controlPointNumber = this->controlPointImage->nx *
                             this->controlPointImage->ny * this->controlPointImage->nz;
    float* message=new float[controlPointNumber*label_num];

    for(int i=0;i<controlPointNumber*label_num;i++){
        this->regularisedCost[i]=this->discretised_measure[i];
        message[i]=0.0;
    }

    for(int i=0;i<label_num;i++){
        cost1[i]=0;
    }

    //calculate mst-cost
    for(int i=(controlPointNumber-1);i>0;i--){ //do for each control point

        int ochild=this->orderedList[i];
        int oparent=this->parentsList[ochild];
        float edgew=this->edgeWeight[ochild];
        float edgew1=1.0f/edgew;

        for(int l=0;l<label_num;l++){
            cost1[l]=this->regularisedCost[ochild*label_num+l]*edgew;
        }

        //fast distance transform see fastDT2.h
        dt3x(cost1,inds,label_len,0,0,0);

        //add mincost to parent node
        for(int l=0;l<label_num;l++){
            message[ochild*label_num+l]=cost1[l]*edgew1;
            this->regularisedCost[oparent*label_num+l]+=cost1[l]*edgew1;
        }
    }

    //backwards pass mst-cost
    for(int i=1;i<controlPointNumber;i++){ //other direction
        int ochild=this->orderedList[i];
        int oparent=this->parentsList[ochild];
        float edgew=this->edgeWeight[this->orderedList[i]];
        float edgew1=1.0f/edgew;

        for(int l=0;l<label_num;l++){
            cost1[l]=(this->regularisedCost[oparent*label_num+l]-message[ochild*label_num+l]+message[oparent*label_num+l])*edgew;
        }

        dt3x(cost1,inds,label_len,0,0,0);
        for(int l=0;l<label_num;l++){
            message[ochild*label_num+l]=cost1[l]*edgew1;
        }

    }

    for(int i=0;i<controlPointNumber*label_num;i++){
        this->orderedList[i]+=message[i];
    }

    //select displacements
    for(int i=0;i<controlPointNumber;i++){
        this->optimalDisplacement[i]=std::min_element(this->regularisedCost+i*label_num,this->regularisedCost+(i+1)*label_num)-(this->regularisedCost+i*label_num);
    }


    delete message;
    delete cost1;
    delete vals;
    delete inds;
}
/*****************************************************/
