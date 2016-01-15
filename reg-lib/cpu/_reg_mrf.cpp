#include "_reg_mrf.h"
/*********************************/
_reg_mrf::_reg_mrf(nifti_image* fixedImage,
                   nifti_image* movingImage,
                   nifti_image* controlPointImage,
                   nifti_image* warpedImage,
                   int label_quant,
                   int label_hw,
                   std::string costMeasureName,
                   float alphaValue)
{
    if(this->fixedImage->nz > 1) {
        dimImage = 3;
    } else {
        dimImage = 2;
    }
    this->fixedImage = fixedImage;
    this->movingImage = movingImage;
    this->controlPointImage = controlPointImage;
    this->label_quant = label_quant;
    this->label_hw = label_hw;
    this->costMeasure = costMeasureName;//not used for the moment
    this->alpha = alphaValue;
    //private variables
    this->grid_step[0] = ceil(controlPointImage->dx / fixedImage->dx);
    this->grid_step[1] = ceil(controlPointImage->dy / fixedImage->dy);
    if(dimImage == 3) {
        this->grid_step[2] = ceil(controlPointImage->dz / fixedImage->dz);
    } else {
        this->grid_step[2] = 1; //HAVE TO CHECK IF IT WORKS IN 2D...
    }
    this->label_len=(label_hw*2+1); //length and total size of displacement space
    this->label_num = pow(label_hw*2+1,dimImage); //|L| number of displacements

    dataCost=new float[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz*label_num];
    regularisedCost=new float[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz*label_num];
    optimalDisplacement=new int[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz];

    fieldLR=new float[controlPointImage->nx*controlPointImage->ny*controlPointImage->nz*dimImage]; //displacement field (on grid level)
    fieldHR=new float[fixedImage->nx*fixedImage->ny*fixedImage->nz*dimImage]; //(and dense voxels) three components for 3D displacement

     this->warpedImage = warpedImage;

}
/*********************************/
_reg_mrf::~_reg_mrf()
{}
/*********************************/
int _reg_mrf::GetLabel_quant()
{
    return label_quant;
}
/*********************************/
int _reg_mrf::GetLabel_hw()
{
    return label_hw;
}
/*********************************/
//int _reg_mrf::GetGrid_step()
//{
//    return grid_step;
//}
/*********************************/
void _reg_mrf::SetLabel_quant(int label_quant)
{
    this->label_quant = label_quant;
}
/*********************************/
void _reg_mrf::SetLabel_hw(int label_hw)
{
    this->label_hw = label_hw;
}
/*********************************/
//void _reg_mrf::SetGrid_step(int grid_step)
//{
//    this->grid_step = grid_step;
//}
/*********************************/
template <class DTYPE>
void _reg_mrf::ComputeSimilarityCost()
{
    int m=fixedImage->nx;
    int n=fixedImage->ny;
    int o=fixedImage->nz; //image dimension
    //
    int m1=controlPointImage->nx;
    int n1=controlPointImage->ny;
    int o1=controlPointImage->nz; //dimensions of grid

    DTYPE* fixedImageData = static_cast<DTYPE> (fixedImage->data);
    //padding of moving image
    DTYPE* movingImageData = static_cast<DTYPE> (movingImage->data);

    int pad1=label_quant*label_hw; //default = 18
    int pad2=pad1*2; //default 18*2=36
    int mp=fixedImage->nx+pad2;
    int np=fixedImage->ny+pad2;
    int op = 1;
    if(dimImage == 3) {
        op=fixedImage->nz+pad2;
    }
    int szp=mp*np*op;
    DTYPE* movingpad=new float[szp*movingImage->nt];
    for(int i=0;i<szp*fixedImage->nt;i++){
        movingpad[i]=0.0f;
    }

    if(dimImage == 3) {
        for(int k=0;k<op;k++){
            for(int j=0;j<np;j++){
                for(int i=0;i<mp;i++){
                    //zero-padding
                    if(i-pad1>=0&&i-pad1<m&&j-pad1>=0&&j-pad1<n&&k-pad1>=0&&k-pad1<o){
                        for(int l1=0;l1<movingImage->nt;l1++){
                            movingpad[l1+(i+j*mp+k*mp*np)*movingImage->nt]=movingImageData[l1+(i-pad1+(j-pad1)*m+(k-pad1)*m*n)*movingImage->nt];
                        }
                    }
                }
            }
        }
    } else {
        for(int j=0;j<np;j++){
            for(int i=0;i<mp;i++){
                //zero-padding
                if(i-pad1>=0&&i-pad1<m&&j-pad1>=0&&j-pad1<n){
                    for(int l1=0;l1<movingImage->nt;l1++){
                        movingpad[l1+(i+j*mp)*movingImage->nt]=movingImageData[l1+(i-pad1+(j-pad1)*m)*movingImage->nt];
                    }
                }
            }
        }
    }

    //skip every other voxel within region of each control point for speed (depending on grid spacing)
    int skipz=1; int skipy=1; const int skipx=1;//2;
    //TO VERIFY -- THIS I DON'T KNOW
    /*
    if(grid_step[0]>4){
        skipy=2; skipz=3; //skipx=2;
    }
    if(grid_step[0]>7){
        skipz=3; skipy=3; //skipx=3;
    }
    if(grid_step[0]==4){
        skipz=2; skipz=2;
    }
    */

    //number of sampling points and adapt alpha to it
    float maxsamp=0;
    if (dimImage == 3) {
        maxsamp=ceil((float)grid_step[0]/(float)skipx)*ceil((float)grid_step[1]/(float)skipy)*ceil((float)grid_step[2]/(float)skipz);
    } else {
        maxsamp=ceil((float)grid_step[0]/(float)skipx)*ceil((float)grid_step[1]/(float)skipy);
    }
    float alphai=(float)grid_step[0]/(alpha*(float)label_quant);
    float alpha1=0.5*alphai/(float)(maxsamp);

    __m128* movingpad128=(__m128*)movingpad;
    __m128* fixed128=(__m128*)fixedImageData;

    if (dimImage == 3) {

#pragma omp parallel for
    for(int z=0;z<o1;z++){ //iterate over all control points
        for(int y=0;y<n1;y++){
            for(int x=0;x<m1;x++){
                //voxel coordinates in fixed image
                int x1=x*grid_step[0];
                int y1=y*grid_step[1];
                int z1=z*grid_step[2];

                for(int l=0;l<label_num;l++){ //iterate over all displacements
                    float out1=0;
                    __m128 dist128={0,0,0,0};

                    //voxel displacements in x,y,z for current l
                    int dz=l/(label_len*label_len);
                    int dy=(l-dz*label_len*label_len)/label_len;
                    int dx=l-dz*label_len*label_len-dy*label_len;
                    dx*=label_quant; dy*=label_quant; dz*=label_quant;
                    //coordinates in moving image
                    int x2=dx+x1; int y2=dy+y1; int z2=dz+z1;
                    //accumulate data cost over voxels within influence region of control point
                    for(int k=0;k<grid_step[2];k+=skipz){
                        for(int j=0;j<grid_step[1];j+=skipy){
                            for(int i=0;i<grid_step[0];i+=skipx){
                                //for(int l1=0;l1<fixedImage->nt/4;l1++){
                                for(int l1=0;l1<fixedImage->nt;l1++){
                                    //coordinates in image space
                                    float t1=fixedImageData[l1+(i+x1+(j+y1)*m+(k+z1)*m*n)*fixedImage->nt];
                                    float t2=movingpad[l1+(i+x2+(j+y2)*mp+(k+z2)*mp*np)*movingImage->nt];
                                    out1+=fabs(t1-t2); //SAD
                                    //__m128 t1=fixed128[l1+(i+x1+(j+y1)*m+(k+z1)*m*n)*fixedImage->nt/4];
                                    //__m128 t2=movingpad128[l1+(i+x2+(j+y2)*mp+(k+z2)*mp*np)*movingImage->nt/4];
                                    //__m128 diff=t1-t2; //SSE speeds-up these distances twofold
                                    //dist128+=_mm_max_ps(-diff,diff);
                                }
                            }
                        }
                    }
                    //horizontal sum of SSE array
                    //float out4[]={0,0,0,0};
                    //_mm_store_ps(out4,dist128);
                    //out1=out4[0]+out4[1]+out4[2]+out4[3];

                    //output matrix (first dimension displacement label, second dim. control point)
                    dataCost[(x+y*m1+z*m1*n1)*label_num+l]=out1*alpha1; //control point coordinates
                }
            }
        }
    }
    } else {
#pragma omp parallel for
        //iterate over all control points
        for(int y=0;y<n1;y++){
            for(int x=0;x<m1;x++){
                //voxel coordinates in fixed image
                int x1=x*grid_step[0];
                int y1=y*grid_step[1];

                for(int l=0;l<label_num;l++){ //iterate over all displacements
                    float out1=0;
                    __m128 dist128={0,0,0,0};

                    //voxel displacements in x,y,z for current l
                    int dy=l/label_len;
                    int dx=l-dy*label_len;

                    dx*=label_quant; dy*=label_quant;
                    //coordinates in moving image
                    int x2=dx+x1; int y2=dy+y1;
                    //accumulate data cost over voxels within influence region of control point
                        for(int j=0;j<grid_step[1];j+=skipy){
                            for(int i=0;i<grid_step[0];i+=skipx){
                                for(int l1=0;l1<fixedImage->nt;l1++){
                                    //coordinates in image space
                                    float t1=fixedImageData[l1+(i+x1+(j+y1)*m)*fixedImage->nt];
                                    float t2=movingpad[l1+(i+x2+(j+y2)*mp)*movingImage->nt];
                                    out1+=fabs(t1-t2); //SAD
                                    //__m128 t1=fixed128[l1+(i+x1+(j+y1)*m+(k+z1)*m*n)*fixedImage->nt/4];
                                    //__m128 t2=movingpad128[l1+(i+x2+(j+y2)*mp+(k+z2)*mp*np)*movingImage->nt/4];
                                    //__m128 diff=t1-t2; //SSE speeds-up these distances twofold
                                    //dist128+=_mm_max_ps(-diff,diff);
                                }
                            }
                        }
                    //horizontal sum of SSE array
                    //float out4[]={0,0,0,0};
                    //_mm_store_ps(out4,dist128);
                    //out1=out4[0]+out4[1]+out4[2]+out4[3];

                    //output matrix (first dimension displacement label, second dim. control point)
                    dataCost[(x+y*m1)*label_num+l]=out1*alpha1; //control point coordinates
                }
            }
        }
    }
    delete movingpad;
    return;
}
/*****************************************************************************************************/
template <class DTYPE>
void _reg_mrf::regularisationMST()
{
    //
    int m1=controlPointImage->nx;
    int n1=controlPointImage->ny;
    int o1=controlPointImage->nz; //dimensions of grid
    int sz1=m1*n1*o1; //number of control points == number of vertices

    int* orderedList=new int[sz1];
    int* parentsList=new int[sz1];
    //weights and indices of potential edges (six per vertex)
    float* edgeWeightMatrix=new float[sz1*dimImage*2];
    int* index_neighbours=new int[sz1*dimImage*2];
    float* edgeWeight=new float[sz1];

    DTYPE* fixedImageData = static_cast<DTYPE> (fixedImage->data);
    edgeGraph(edgeWeightMatrix,index_neighbours,fixedImageData,fixedImage->nx,fixedImage->ny,fixedImage->nz,grid_step);

    primsMST(orderedList,parentsList,edgeWeight,edgeWeightMatrix,index_neighbours,m1,n1,o1);
    delete index_neighbours; delete edgeWeightMatrix;

    regularisation(regularisedCost,optimalDisplacement,dataCost,orderedList,parentsList,edgeWeight,label_hw,m1,n1,o1);

    for(int i=0;i<sz1;i++){
        //simply select argmin
        int l=std::min_element(regularisedCost+i*label_num,regularisedCost+(i+1)*label_num)-(regularisedCost+i*label_num);
        //voxel displacements in x,y,z for current l
        int dz=l/(label_len*label_len);
        int dy=(l-dz*label_len*label_len)/label_len;
        int dx=l-dz*label_len*label_len-dy*label_len;
        dx-=label_hw; dy-=label_hw; dz-=label_hw;
        dx*=label_quant; dy*=label_quant; dz*=label_quant;
        fieldLR[i]=dx; fieldLR[i+sz1]=dy; fieldLR[i+sz1*2]=dz;

    }

    delete orderedList; delete parentsList; delete edgeWeight;
}
/********************************************************************************************************/
//input lowres field (defined for control point), output non-parametric vectors
void _reg_mrf::upsampleDisplacements()
{
    int m1=controlPointImage->nx;
    int n1=controlPointImage->ny;
    int o1=controlPointImage->nz; //dimensions of grid
    int sz1=m1*n1*o1; //number LR control points
    int m=fixedImage->nx;
    int n=fixedImage->ny;
    int o=fixedImage->nz;
    int sz=m*n*o; //number HR voxels
    //scaling parameters for coordinates
    float scale_m=(float)m/(float)m1;
    float scale_n=(float)n/(float)n1;
    float scale_o=(float)o/(float)o1;
    //interpolation coordinates
    float* x1=new float[sz];
    float* y1=new float[sz];
    float* z1=new float[sz];
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                x1[i+j*m+k*m*n]=i/scale_n;
                y1[i+j*m+k*m*n]=j/scale_m;
                z1[i+j*m+k*m*n]=k/scale_o;
            }
        }
    }
    //simple linear interpolation
    interp3(fieldHR,fieldLR,x1,y1,z1,m,n,o,m1,n1,o1,false);
    interp3(fieldHR+sz,fieldLR+sz1,x1,y1,z1,m,n,o,m1,n1,o1,false);
    interp3(fieldHR+2*sz,fieldLR+2*sz1,x1,y1,z1,m,n,o,m1,n1,o1,false);

    delete x1;
    delete y1;
    delete z1;

}
/********************************************************************************************************/
template <class DTYPE>
void _reg_mrf::warpMovingImage()
{
    DTYPE* movingImageData = static_cast<DTYPE> (movingImage->data);
    DTYPE* warpedImageData = static_cast<DTYPE> (warpedImage->data);

    int m=fixedImage->nx;
    int n=fixedImage->ny;
    int o=fixedImage->nz; //image dimension
    int sz=m*n*o; //number of voxels

    interp3(warpedImageData,movingImageData,fieldHR,fieldHR+sz,fieldHR+2*sz,m,n,o,m,n,o,true); //true->plus identity
}
/********************************************************************************************************/
/*
void _reg_mrf::Run()
{
    this->ComputeSimilarityCost();
    this->regularisationMST();
    this->upsampleDisplacements();
    this->warpMovingImage();
}
*/
/********************************************************************************************************/
//returns list of potential edges (indicated by grid-point indices of neighbours) and their weights
void edgeGraph(float* edgeWeightMatrix,int* index_neighbours,float* fixed,int m,int n,int o,int* grid_step){

    int sz=m*n*o; //image size
    int m1=m/grid_step[0];
    int n1=n/grid_step[1];
    int o1=o/grid_step[2]; //dimensions of grid
    if(o == 1) {
        o1=1;
    }
    int num_vertices=m1*n1*o1; //number of control points = number of vertices

    int num_neighbours=6;
    if(o == 1) {
        num_neighbours=4;
    }
    for(int i=0;i<num_vertices*num_neighbours;i++){
        edgeWeightMatrix[i]=0.0;
        index_neighbours[i]=-1;
    }
    ////////////////////////////////////////////
    if(o>1) {
    //standard six-neighbourhood for grid graph
    int dx[6]={-1,1,0,0,0,0};
    int dy[6]={0,0,-1,1,0,0};
    int dz[6]={0,0,0,0,-1,1};
    //calculate edge-weights based on SAD of groups of voxels (for each control-point)
    for(int z=0;z<o1;z++){
        for(int y=0;y<n1;y++){
            for(int x=0;x<m1;x++){ //for every grid point
                for(int nb=0;nb<num_neighbours;nb++){ //for six neighbours
                    //only add neighbours if within image domain
                    if((x+dx[nb])>=0&(x+dx[nb])<m1&(y+dy[nb])>=0&(y+dy[nb])<n1&(z+dz[nb])>=0&(z+dz[nb])<o1){
                        index_neighbours[x+y*m1+z*m1*n1+nb*num_vertices]=x+dx[nb]+(y+dy[nb])*m1+(z+dz[nb])*m1*n1;
                        //for all voxels within block around grid point
                        for(int z1=0;z1<grid_step[2];z1++){
                            for(int y1=0;y1<grid_step[1];y1++){
                                for(int x1=0;x1<grid_step[0];x1++){
                                    int xx=x*grid_step[0]+x1;
                                    int yy=y*grid_step[1]+y1;
                                    int zz=z*grid_step[2]+z1;
                                    int xx2=(x+dx[nb])*grid_step[0]+x1;
                                    int yy2=(y+dy[nb])*grid_step[1]+y1;
                                    int zz2=(z+dz[nb])*grid_step[2]+z1;
                                    edgeWeightMatrix[x+y*m1+z*m1*n1+nb*num_vertices]+=fabs(fixed[xx+yy*m+zz*m*n]-fixed[xx2+yy2*m+zz2*m*n]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    } else {
        //standard four-neighbourhood for grid graph
        int dx[4]={-1,1,0,0};
        int dy[4]={0,0,-1,1};
        //calculate edge-weights based on SAD of groups of voxels (for each control-point)
            for(int y=0;y<n1;y++){
                for(int x=0;x<m1;x++){ //for every grid point
                    for(int nb=0;nb<num_neighbours;nb++){ //for six neighbours
                        //only add neighbours if within image domain
                        if((x+dx[nb])>=0&(x+dx[nb])<m1&(y+dy[nb])>=0&(y+dy[nb])<n1){
                            index_neighbours[x+y*m1+nb*num_vertices]=x+dx[nb]+(y+dy[nb])*m1;
                            //for all voxels within block around grid point
                                for(int y1=0;y1<grid_step[1];y1++){
                                    for(int x1=0;x1<grid_step[0];x1++){
                                        int xx=x*grid_step[0]+x1;
                                        int yy=y*grid_step[1]+y1;
                                        int xx2=(x+dx[nb])*grid_step[0]+x1;
                                        int yy2=(y+dy[nb])*grid_step[1]+y1;
                                        edgeWeightMatrix[x+y*m1+nb*num_vertices]+=fabs(fixed[xx+yy*m]-fixed[xx2+yy2*m]);
                                    }
                                }
                        }
                    }
                }
            }
    }
    //normalise edgeweights by stddev of image
    float meanim=0.0;
    for(int i=0;i<sz;i++){
        meanim+=fixed[i];
    }
    meanim/=(float)(sz);
    float stdim=0.0;
    for(int i=0;i<sz;i++){
        stdim+=pow(fixed[i]-meanim,2);
    }
    stdim=sqrt(stdim/(float)(sz));

    for(int i=0;i<num_vertices*num_neighbours;i++){
        edgeWeightMatrix[i]/=(float)pow(grid_step[0],3);//2 in 2-D ?
    }
    for(int i=0;i<num_vertices*num_neighbours;i++){
        edgeWeightMatrix[i]=-exp(-edgeWeightMatrix[i]/(2.0f*stdim));
    }
}
/********************************************************************************************************/
//extract minimum-spanning-tree from edge-weights
//edges with large intensity difference are more likely to be cut
void primsMST(int* orderedList,int* parentsList,float* edgeWeight,float* edgeWeightMatrix,int* index_neighbours,int m, int n, int o){

    int num_vertices = m*n*o;
    int currentNode=0; //arbritary root node
    //list of nodes already in MST
    bool* addedToMST=new bool[num_vertices];
    for(int i=0;i<num_vertices;i++){
        addedToMST[i]=false;
    }
    addedToMST[currentNode]=true;
    std::pair<short,int>* treeLevel=new std::pair<short,int>[num_vertices];
    treeLevel[currentNode]={0,currentNode};

    int num_neighbours=6;
    if(o==1) {
        num_neighbours=4;
    }

    parentsList[currentNode]=-1; //root has no parent
    std::priority_queue<Edge> priority; //priority queue

    float mincost=0.0f;
    //run n-1 times so that all nodes added
    for(int i=0;i<num_vertices-1;i++){
        //add edges of new node to priority queue
        for(int j=0;j<num_neighbours;j++){
            int index_j=index_neighbours[currentNode+j*num_vertices];
            float weight=edgeWeightMatrix[currentNode+j*num_vertices];
            if(index_j>=0){
                priority.push({weight,currentNode,index_j});
            }

        }
        currentNode=-1;
        while(currentNode==-1){
            Edge bestEdge=priority.top();
            priority.pop();
            //test whether endIndex of edge is already in MST
            if(addedToMST[bestEdge.startIndex]&&not(addedToMST[bestEdge.endIndex])){
                mincost+=-bestEdge.weight;

                edgeWeight[bestEdge.endIndex]=-bestEdge.weight;

                currentNode=bestEdge.endIndex;
                addedToMST[bestEdge.endIndex]=true;
                parentsList[bestEdge.endIndex]=bestEdge.startIndex;
                treeLevel[bestEdge.endIndex]={treeLevel[bestEdge.startIndex].first+1,bestEdge.endIndex};

            }

        }

    }
    //generate list of nodes ordered by tree depth
    std::sort(treeLevel,treeLevel+num_vertices);
    //printf("max tree depth: %d, mincost: %f\n",treeLevel[num_vertices-1].first,mincost);
    for(int i=0;i<num_vertices;i++){
        orderedList[i]=treeLevel[i].second;
    }
}
/************************************************************************************************************************************/
void regularisation(float* marginals,int* selected,float* dataCost,int* ordered,int* parents,float* edgeweights,int label_hw,int m1,int n1, int o1){

    int dim = 3;
    if (o1 > 1) {
        dim = 2;
    }
    int sz1 = m1*n1*o1;
    //dense displacement space
    int label_len=(label_hw*2+1); //length and total size of displacement space
    int label_num=pow((label_hw*2+1),dim);

    //buffer variable
    float *cost1=new float[label_num];
    float *vals=new float[label_num];
    int *inds=new int[label_num];

    //array of messages
    float* message=new float[sz1*label_num];

    for(int i=0;i<sz1*label_num;i++){
        marginals[i]=dataCost[i];
        message[i]=0.0;
    }


    for(int i=0;i<label_num;i++){
        cost1[i]=0;
    }

    //calculate mst-cost
    for(int i=(sz1-1);i>0;i--){ //do for each control point

        int ochild=ordered[i];
        int oparent=parents[ochild];
        float edgew=edgeweights[ordered[i]];
        float edgew1=1.0f/edgew;

        for(int l=0;l<label_num;l++){
            cost1[l]=marginals[ochild*label_num+l]*edgew;
        }

        //fast distance transform see fastDT2.h
        dt3x(cost1,inds,label_len,0,0,0);

        //add mincost to parent node
        for(int l=0;l<label_num;l++){
            message[ochild*label_num+l]=cost1[l]*edgew1;
            marginals[oparent*label_num+l]+=cost1[l]*edgew1;

        }

    }

    //backwards pass mst-cost
    for(int i=1;i<sz1;i++){ //other direction
        int ochild=ordered[i];
        int oparent=parents[ochild];
        float edgew=edgeweights[ordered[i]];
        float edgew1=1.0f/edgew;

        for(int l=0;l<label_num;l++){
            cost1[l]=(marginals[oparent*label_num+l]-message[ochild*label_num+l]+message[oparent*label_num+l])*edgew;
        }

        dt3x(cost1,inds,label_len,0,0,0);
        for(int l=0;l<label_num;l++){
            message[ochild*label_num+l]=cost1[l]*edgew1;
        }

    }

    for(int i=0;i<sz1*label_num;i++){
        marginals[i]+=message[i];
    }

    //select displacements
    for(int i=0;i<sz1;i++){
        selected[i]=std::min_element(marginals+i*label_num,marginals+(i+1)*label_num)-(marginals+i*label_num);

    }


    delete message;
    delete cost1;
    delete vals;
    delete inds;


}
/**********************************************************************************************/
//fast distance transform for message computation following Pedro Felzenszwalb's implementation
//see http://cs.brown.edu/~pff/dt/index.html for details
void dt1sq(float *val,int* ind,int len,float offset,int k,int* v,float* z,float* f,int* ind1){
    float INF=1e10;
    int j=0;
    z[0]=-INF;
    z[1]=INF;
    v[0]=0;
    for(int q=1;q<len;q++){
        float s=((val[q*k]+q*q)-(val[v[j]*k]+v[j]*v[j]))/(2.0*(q-v[j]));

        while(s<=z[j]){
            j--;
            s=((val[q*k]+q*q)-(val[v[j]*k]+v[j]*v[j]))/(2.0*(q-v[j]));
        }

        j++;
        v[j]=q;
        z[j]=s;
        z[j+1]=INF;

    }
    for(int q=0;q<len;q++){
        f[q]=val[q*k]; //needs to be added to fastDT2 otherwise incorrect
        ind1[q]=ind[q*k];
    }

    j=0;
    for(int q=0;q<len;q++){
        while(z[j+1]<(q-offset)){  //was wrong -offset is now correct
            j++;
        }
        ind[q*k]=ind1[v[j]];
        val[q*k]=(q-offset-v[j])*(q-offset-v[j])+f[v[j]];
    }
}
/**********************************************************************************************/
void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz){
    //rl is length of one side
    for(int i=0;i<rl*rl*rl;i++){
        indr[i]=i;
    }
    int* v=new int[rl]; //slightly faster if not intitialised in each loop
    float* z=new float[rl+1];
    float* f=new float[rl];
    int* i1=new int[rl];

    for(int k=0;k<rl;k++){
        for(int i=0;i<rl;i++){
            dt1sq(r+i+k*rl*rl,indr+i+k*rl*rl,rl,-dx,rl,v,z,f,i1);
        }
    }

    for(int k=0;k<rl;k++){
        for(int j=0;j<rl;j++){
            dt1sq(r+j*rl+k*rl*rl,indr+j*rl+k*rl*rl,rl,-dy,1,v,z,f,i1);//);
        }
    }

    for(int j=0;j<rl;j++){
        for(int i=0;i<rl;i++){
            dt1sq(r+i+j*rl,indr+i+j*rl,rl,-dz,rl*rl,v,z,f,i1);//);
        }
    }
    float min1=*std::min_element(r,r+rl*rl*rl);
    for(int i=0;i<rl*rl*rl;i++){
        r[i]-=min1;
    }
    delete []i1;
    delete []f;

    delete []v;
    delete []z;


}
/*******************************************************************************************/
void interp3(float* interp,float* input,float* x1,float* y1,float* z1,int m,int n,int o,int m2,int n2,int o2,bool flag){
    //m,n,o are output dimensions, m2,n2,o2 are input dimensions
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                int x=floor(x1[i+j*m+k*m*n]); int y=floor(y1[i+j*m+k*m*n]);  int z=floor(z1[i+j*m+k*m*n]);
                float dx=x1[i+j*m+k*m*n]-x; float dy=y1[i+j*m+k*m*n]-y; float dz=z1[i+j*m+k*m*n]-z;
                if(flag){
                    x+=i; y+=j; z+=k; //plus identity transform
                }
                if(x>=0&&x<m2&&y>=0&&y<n2&&z>=0&&z<o2){
                    interp[i+j*m+k*m*n]=(1.0-dx)*(1.0-dy)*(1.0-dz)*input[x+y*m2+z*m2*n2]+dx*(1.0-dy)*(1.0-dz)*input[std::min(x+1,m2-1)+y*m2+z*m2*n2]+
                    (1.0-dx)*dy*(1.0-dz)*input[x+std::min(y+1,n2-1)*m2+z*m2*n2]+(1.0-dx)*(1.0-dy)*dz*input[x+y*m2+std::min(z+1,o2-1)*m2*n2]+
                    dx*dy*(1.0-dz)*input[std::min(x+1,m2-1)+std::min(y+1,n2-1)*m2+z*m2*n2]+dx*(1.0-dy)*dz*input[std::min(x+1,m2-1)+y*m2+std::min(z+1,o2-1)*m2*n2]+
                    (1.0-dx)*dy*dz*input[x+std::min(y+1,n2-1)*m2+std::min(z+1,o2-1)*m2*n2]+dx*dy*dz*input[std::min(x+1,m2-1)+(y+1)*m2+std::min(z+1,o2-1)*m2*n2];
                }
                else{
                    interp[i+j*m+k*m*n]=0.0f;
                }
            }
        }
    }
}
