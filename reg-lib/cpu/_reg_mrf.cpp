#include "_reg_mrf.h"

/*****************************************************/
reg_mrf::reg_mrf(reg_measure *_measure,
                 nifti_image *_referenceImage,
                 nifti_image *_controlPointImage,
                 int _discrete_radius,
                 int _discrete_increment,
                 float _reg_weight)
{
   this->measure = _measure;
   this->referenceImage = _referenceImage;
   this->controlPointImage = _controlPointImage;
   this->discrete_radius = _discrete_radius;
   this->discrete_increment = _discrete_increment;
   this->regularisation_weight = _reg_weight;

   this->image_dim = this->referenceImage->nz > 1 ? 3 :2;
   this->label_1D_num = (this->discrete_radius / this->discrete_increment ) * 2 + 1;
   this->label_nD_num = std::pow(this->label_1D_num,this->image_dim);
   this->node_number = (size_t)this->controlPointImage->nx *
         this->controlPointImage->ny * this->controlPointImage->nz;

   // Allocate the discretised values in voxel
   int *discrete_values_vox = (int *)malloc(this->label_1D_num*sizeof(int));
   int currentValue = -this->discrete_radius;
   for(int i = 0;i<this->label_1D_num;i++) {
      discrete_values_vox[i]=currentValue;
      currentValue+=this->discrete_increment;
   }

   // Allocate the discretised values in millimeter
   this->discrete_values_mm = (float **)malloc(this->image_dim*sizeof(float *));
   for(int i=0;i<this->image_dim;++i){
       this->discrete_values_mm[i] = (float *)malloc(this->label_nD_num*sizeof(float));
   }
   float disp_vox[3];
   mat44 vox2mm = this->referenceImage->qto_xyz;
   if(this->referenceImage->sform_code>0)
      vox2mm = this->referenceImage->sto_xyz;
   int i=0;
   for(int z=0; z<this->label_1D_num; ++z){
      disp_vox[2]=discrete_values_vox[z];
      for(int y=0; y<this->label_1D_num; ++y){
         disp_vox[1]=discrete_values_vox[y];
         for(int x=0; x<this->label_1D_num; ++x){
            disp_vox[0]=discrete_values_vox[x];
            this->discrete_values_mm[0][i] =
                  disp_vox[0] * vox2mm.m[0][0] +
                  disp_vox[1] * vox2mm.m[0][1] +
                  disp_vox[2] * vox2mm.m[0][2];
            this->discrete_values_mm[1][i] =
                  disp_vox[0] * vox2mm.m[1][0] +
                  disp_vox[1] * vox2mm.m[1][1] +
                  disp_vox[2] * vox2mm.m[1][2];
            this->discrete_values_mm[2][i] =
                  disp_vox[0] * vox2mm.m[2][0] +
                  disp_vox[1] * vox2mm.m[2][1] +
                  disp_vox[2] * vox2mm.m[2][2];
            ++i;
         }
      }
   }
   free(discrete_values_vox);


   //To store the cost data term - originaly SAD between images.
   this->discretised_measures = (float *)malloc(this->node_number*this->label_nD_num*sizeof(float));

   // Allocate the arrays to store the tree
   this->orderedList = (int *) malloc(this->node_number*sizeof(int));
   this->parentsList = (int *) malloc(this->node_number*sizeof(int));
   this->edgeWeight = (float *) malloc(this->node_number*sizeof(float));

   //regulatization - optimization
   this->regularised_cost= (float *)malloc(this->node_number*this->label_nD_num*sizeof(float));
   this->optimal_label_index=(int *)malloc(this->node_number*sizeof(int));

   this->initialised = false;
}
/*****************************************************/
reg_mrf::~reg_mrf()
{
   if(this->discretised_measures!=NULL)
      free(this->discretised_measures);
   this->discretised_measures=NULL;

   if(this->orderedList!=NULL)
      free(this->orderedList);
   this->orderedList=NULL;

   if(this->parentsList!=NULL)
      free(this->parentsList);
   this->parentsList=NULL;

   if(this->edgeWeight!=NULL)
      free(this->edgeWeight);
   this->edgeWeight=NULL;

   if(this->regularised_cost!=NULL)
      free(this->regularised_cost);
   this->regularised_cost=NULL;

   if(this->optimal_label_index!=NULL)
      free(this->optimal_label_index);
   this->optimal_label_index=NULL;

   for(int i=0; i<this->image_dim; ++i){
      if(this->discrete_values_mm[i]!=NULL)
         free(this->discrete_values_mm[i]);
      this->discrete_values_mm[i]=NULL;
   }
   if(this->discrete_values_mm!=NULL)
      free(this->discrete_values_mm);
   this->discrete_values_mm=NULL;
}
/*****************************************************/
void reg_mrf::Initialise()
{
   // Create the minimum spamming tree
   int edge_number = this->node_number*this->image_dim*2;
   float *edgeWeightMatrix = (float *)calloc(edge_number,sizeof(float));
   int *index_neighbours = (int *)malloc(edge_number*sizeof(int));
   for(int i =0;i<edge_number;i++) {
      index_neighbours[i]=-1;
   }
   this->GetGraph(edgeWeightMatrix, index_neighbours);
   this->GetPrimsMST(edgeWeightMatrix, index_neighbours);
   free(edgeWeightMatrix);
   free(index_neighbours);
   this->initialised = true;
#ifndef NDEBUG
   reg_print_msg_debug("reg_mrf::Initilisation done.");
#endif
}
/*****************************************************/
float* reg_mrf::GetDiscretisedMeasurePtr()
{
   return this->discretised_measures;
}
/*****************************************************/
void reg_mrf::GetDiscretisedMeasure()
{
   measure->GetDiscretisedValue(this->controlPointImage,
                                this->discretised_measures,
                                this->discrete_radius,
                                this->discrete_increment);
 #ifndef NDEBUG
   reg_print_msg_debug("reg_mrf::GetDiscretisedMeasure done");
#endif
}
/*****************************************************/
void reg_mrf::getOptimalLabel()
{
   for(int node=0; node<this->node_number; ++node) {
      this->optimal_label_index[node]=
         std::min_element(this->regularised_cost+node*this->label_nD_num,this->regularised_cost+(node+1)*this->label_nD_num) -
         (this->regularised_cost+node*this->label_nD_num);
   }
}
/*****************************************************/
void reg_mrf::UpdateNodePositions()
{
   //Update the control point position
   float *cpPtrX = static_cast<float *>(this->controlPointImage->data);
   float *cpPtrY = &cpPtrX[this->node_number];
   float *cpPtrZ = &cpPtrY[this->node_number];

   size_t voxel=0;
   for(int z=0; z<this->controlPointImage->nz; z++) {
      for(int y=0; y<this->controlPointImage->ny; y++) {
         for(int x=0; x<this->controlPointImage->nx; x++) {
            int optimal_id = this->optimal_label_index[voxel];
            cpPtrX[voxel] += this->discrete_values_mm[0][optimal_id];
            cpPtrY[voxel] += this->discrete_values_mm[1][optimal_id];
            cpPtrZ[voxel] += this->discrete_values_mm[2][optimal_id];
            ++voxel;
         }
      }
   }
#ifndef NDEBUG
  reg_print_msg_debug("reg_mrf::Optimise done");
#endif
}
/*****************************************************/
void reg_mrf::Run()
{
   if(this->initialised==false)
      this->Initialise();
   // Compute the discretised data term values
   this->GetDiscretisedMeasure();
   // Compute the regularisation term
   //for(int i=0;i<100; ++i){
       this->GetRegularisation();
       // Extract the best label
       //memcpy(this->regularised_cost, this->discretised_measures, this->node_number*this->label_nD_num*sizeof(float));
       this->getOptimalLabel();
       // Update the control point positions
       this->UpdateNodePositions();
   //}
}
/*****************************************************/
/*****************************************************/
template <class DTYPE>
void GetGraph_core3D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     int* index_neighbours,
                     nifti_image *refImage,
                     int *mask)
{
   int cpx, cpy, cpz, t, x, y, z, blockIndex, voxIndex, voxIndex_t;
   float gridVox[3], imageVox[3];
   // Define the transformation matrices
   mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
   if(controlPointGridImage->sform_code>0)
      grid_vox2mm = &controlPointGridImage->sto_xyz;
   mat44 *image_mm2vox = &refImage->qto_ijk;
   if(refImage->sform_code>0)
      image_mm2vox = &refImage->sto_ijk;
   mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

   size_t node_number = (size_t)controlPointGridImage->nx *
         controlPointGridImage->ny * controlPointGridImage->nz;

   // Compute the block size
   int blockSize[3]={
      (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
      (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
      (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
   };
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
      for(cpy=0; cpy<controlPointGridImage->ny; ++cpy){
         for(cpx=0; cpx<controlPointGridImage->nx; ++cpx){
            //DEBUG
            //int sz=refImage->nx * refImage->ny * refImage->nz;
            //int m=refImage->nx;
            //int n=refImage->ny;
            //int o=refImage->nz;
            //int grid_step = blockSize[0];
            //int m1=m/grid_step;
            //int n1=n/grid_step;
            //int o1=o/grid_step;
            //int num_vertices = m1*n1*o1;
            //for(cpz=0; cpz<o1; ++cpz){
            //        for(cpy=0; cpy<n1; ++cpy){
            //            for(cpx=0; cpx<m1; ++cpx){
            //DEBUG
            //Because I reuse this variable after.
            gridVox[2] = cpz;
            gridVox[1] = cpy;
            gridVox[0] = cpx;
            // Compute the corresponding image voxel position
            reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
            imageVox[0]=reg_round(imageVox[0]);
            imageVox[1]=reg_round(imageVox[1]);
            imageVox[2]=reg_round(imageVox[2]);
            //DEBUG
            //imageVox[0]=gridVox[0]*controlPointGridImage->dx / refImage->dx;
            //imageVox[1]=gridVox[1]*controlPointGridImage->dy / refImage->dy;
            //imageVox[2]=gridVox[2]*controlPointGridImage->dz / refImage->dz;
            //DEBUG
            // Extract the block in the reference image
            blockIndex = 0;
            for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
               for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                  for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                     //DEBUG
                     //for(z=imageVox[2]; z<imageVox[2]+blockSize[2]; ++z){
                     //    for(y=imageVox[1]; y<imageVox[1]+blockSize[1]; ++y){
                     //        for(x=imageVox[0]; x<imageVox[0]+blockSize[0]; ++x){
                     //DEBUG
                     if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                        voxIndex = x+y*refImage->nx+z*refImage->nx*refImage->ny;
                        if(mask[voxIndex]>-1){
                           for(t=0; t<refImage->nt; ++t){
                              voxIndex_t = voxIndex+t*refImage->nx*refImage->ny*refImage->nz;
                              refBlockValue[blockIndex] = refImgPtr[voxIndex_t];
                              blockIndex++;
                           } //t
                        }
                     } else {
                        for(t=0; t<refImage->nt; ++t){
                           refBlockValue[blockIndex] = 0.0;
                           blockIndex++;
                        }
                     }
                  } // x
               } // y
            } // z
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
               if(gridVox[0]>=0 && gridVox[0]<controlPointGridImage->nx &&
                     gridVox[1]>=0 && gridVox[1]<controlPointGridImage->ny &&
                     gridVox[2]>=0 && gridVox[2]<controlPointGridImage->nz) {
                  //DEBUG
                  //if(gridVox[0]>=0 && gridVox[0]<m1 &&
                  //   gridVox[1]>=0 && gridVox[1]<n1 &&
                  //   gridVox[2]>=0 && gridVox[2]<o1) {
                  //DEBUG
                  // Compute the corresponding image voxel position
                  reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                  imageVox[0]=reg_round(imageVox[0]);
                  imageVox[1]=reg_round(imageVox[1]);
                  imageVox[2]=reg_round(imageVox[2]);
                  //DEBUG
                  //imageVox[0]=gridVox[0]*controlPointGridImage->dx / refImage->dx;
                  //imageVox[1]=gridVox[1]*controlPointGridImage->dy / refImage->dy;
                  //imageVox[2]=gridVox[2]*controlPointGridImage->dz / refImage->dz;
                  //DEBUG
                  if(imageVox[0]>-1 && imageVox[0]<refImage->nx &&
                        imageVox[1]>-1 && imageVox[1]<refImage->ny &&
                        imageVox[2]>-1 && imageVox[2]<refImage->nz) {
                     blockIndex = 0;
                     for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                        for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                           for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                              //DEBUG
                              //for(z=imageVox[2]; z<imageVox[2]+blockSize[2]; ++z){
                              //    for(y=imageVox[1]; y<imageVox[1]+blockSize[1]; ++y){
                              //        for(x=imageVox[0]; x<imageVox[0]+blockSize[0]; ++x){
                              //DEBUG
                              if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                                 voxIndex = x+y*refImage->nx+z*refImage->nx*refImage->ny;
                                 if(mask[voxIndex]>-1){
                                    for(t=0; t<refImage->nt; ++t){
                                       voxIndex_t = voxIndex+t*refImage->nx*refImage->ny*refImage->nz;
                                       neighbourBlockValue[blockIndex] = refImgPtr[voxIndex_t];
                                       blockIndex++;
                                    } //t
                                 }
                              }else {
                                 for(t=0; t<refImage->nt; ++t){
                                    neighbourBlockValue[blockIndex] = 0.0;
                                    blockIndex++;
                                 } //t
                              }
                           } // x
                        } // y
                     } // z

                     SADNeighbourValue = 0;
                     for(int sadIndex=0;sadIndex<voxelBlockNumber;sadIndex++) {
                        SADNeighbourValue += std::abs(neighbourBlockValue[sadIndex]-refBlockValue[sadIndex]);
                     }
                     if(SADNeighbourValue == 0) {
                         SADNeighbourValue = std::numeric_limits<float>::epsilon();
                     }
                     //store results:
                     index_neighbours[cpx+cpy*controlPointGridImage->nx+
                           cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                           ngh_index*node_number]=
                           cpx+dx[ngh_index]+(cpy+dy[ngh_index])*controlPointGridImage->nx+
                           (cpz+dz[ngh_index])*controlPointGridImage->nx*controlPointGridImage->ny;
                     edgeWeightMatrix[cpx+cpy*controlPointGridImage->nx+
                           cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                           ngh_index*node_number]=SADNeighbourValue;
                     //DEBUG
                     //index_neighbours[cpx+cpy*m1+
                     //        cpz*m1*n1+
                     //        ngh_index*num_vertices]=
                     //        cpx+dx[ngh_index]+(cpy+dy[ngh_index])*m1+
                     //        (cpz+dz[ngh_index])*m1*n1;
                     //edgeWeightMatrix[cpx+cpy*m1+
                     //        cpz*m1*n1+
                     //        ngh_index*num_vertices]=SADNeighbourValue;
                     //DEBUG
                  } else {
                     //store results:
                     index_neighbours[cpx+cpy*controlPointGridImage->nx+
                           cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                           ngh_index*node_number]=
                           cpx+dx[ngh_index]+(cpy+dy[ngh_index])*controlPointGridImage->nx+
                           (cpz+dz[ngh_index])*controlPointGridImage->nx*controlPointGridImage->ny;

                     edgeWeightMatrix[cpx+cpy*controlPointGridImage->nx+
                           cpz*controlPointGridImage->nx*controlPointGridImage->ny+
                           ngh_index*node_number]=0.0;
                     //DEBUG
                     //index_neighbours[cpx+cpy*m1+
                     //        cpz*m1*n1+
                     //        ngh_index*num_vertices]=
                     //        cpx+dx[ngh_index]+(cpy+dy[ngh_index])*m1+
                     //        (cpz+dz[ngh_index])*m1*n1;
                     //edgeWeightMatrix[cpx+cpy*m1+
                     //        cpz*m1*n1+
                     //        ngh_index*num_vertices]=0.0;
                     //DEBUG
                  }
               }
            }
         } //cpx
      } //cpy
   } //cpz
   //
   //
   //normalise edgeweights by stddev of image ???????
   float stdim=reg_tools_getSTDValue(refImage);

   for(int i=0;i<node_number*6;i++){
      edgeWeightMatrix[i]/=voxelBlockNumber;
   }
   for(int i=0;i<node_number*6;i++){
      edgeWeightMatrix[i]=-exp(-edgeWeightMatrix[i]/(2.0f*stdim));
   }
   //DEBUG
   //for(int i=0;i<num_vertices*6;i++){
   //    edgeWeightMatrix[i]/=voxelBlockNumber;
   //    }
   //for(int i=0;i<num_vertices*6;i++){
   //    edgeWeightMatrix[i]=-exp(-edgeWeightMatrix[i]/(2.0f*stdim));
   //    }
   //DEBUG
}
/* *************************************************************** */
template <class DTYPE>
void GetGraph_core2D(nifti_image* controlPointGridImage,
                     float* edgeWeightMatrix,
                     int* index_neighbours,
                     nifti_image *refImage,
                     int *mask)
{
   reg_print_fct_warn("GetGraph_core2D");
   reg_print_msg_warn("No yet implemented");
   reg_exit();
}
/* *************************************************************** */
void reg_mrf::GetGraph(float *edgeWeightMatrix, int *index_neighbours)
{
   if(this->referenceImage->nz > 1) {
      switch(this->referenceImage->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         GetGraph_core3D<float>
               (this->controlPointImage,
                edgeWeightMatrix,
                index_neighbours,
                this->referenceImage,
                this->measure->GetReferenceMask()
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         GetGraph_core3D<double>
               (this->controlPointImage,
                edgeWeightMatrix,
                index_neighbours,
                this->referenceImage,
                this->measure->GetReferenceMask()
                );
         break;
      default:
         reg_print_fct_error("reg_mrf::GetGraph");
         reg_print_msg_error("Unsupported datatype");
         reg_exit();
      }
   } else {
      switch(this->referenceImage->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         GetGraph_core2D<float>
               (this->controlPointImage,
                edgeWeightMatrix,
                index_neighbours,
                this->referenceImage,
                this->measure->GetReferenceMask()
                );
         break;
      case NIFTI_TYPE_FLOAT64:
         GetGraph_core2D<double>
               (this->controlPointImage,
                edgeWeightMatrix,
                index_neighbours,
                this->referenceImage,
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
/* *************************************************************** */
/*****************************************************/
//CUT THE EDGES WITH HIGH COST = INTENSITY DIFFERENCES!
/*****************************************************/
void reg_mrf::GetPrimsMST(float *edgeWeightMatrix,
                          int *index_neighbours)
{
   int num_vertices = this->controlPointImage->nx *
         this->controlPointImage->ny * this->controlPointImage->nz;
   //DEBUG
   //int blockSize[3]={
   //    (int)reg_ceil(controlPointImage->dx / referenceImage->dx),
   //    (int)reg_ceil(controlPointImage->dy / referenceImage->dy),
   //    (int)reg_ceil(controlPointImage->dz / referenceImage->dz),
   //};
   //int sz=referenceImage->nx * referenceImage->ny * referenceImage->nz;
   //int m=referenceImage->nx;
   //int n=referenceImage->ny;
   //int o=referenceImage->nz;
   //int grid_step = blockSize[0];
   //int m1=m/grid_step;
   //int n1=n/grid_step;
   //int o1=o/grid_step;
   //num_vertices = m1*n1*o1;
   //DEBUG
   int currentNode=0; //arbritary root node
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
      //add edges of new node to priority queue
      for(int j=0;j<num_neighbours;j++){
         int index_j=index_neighbours[currentNode+j*num_vertices];
         float weight=edgeWeightMatrix[currentNode+j*num_vertices];
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
            mincost+=-bestEdge.weight; //if normalization by -exp
            //mincost+=bestEdge.weight;

            edgeWeight[bestEdge.endIndex]=-bestEdge.weight;//if normalization by -exp
            //this->edgeWeight[bestEdge.endIndex]=bestEdge.weight;

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
   delete treeLevel;
   delete addedToMST;
}
/*****************************************************/
void reg_mrf::GetRegularisation()
{
   /* Incremental diffusion regularisation of parametrised transformation
     using (globally optimal) belief-propagation on minimum spanning tree.
     Fast distance transform uses squared differences.
     Similarity cost for each node and label has to be given as input.
    */

   //buffer variable
   float *cost1=new float[this->label_nD_num];
   float *vals=new float[this->label_nD_num];
   int *inds=new int[this->label_nD_num];

   //DEBUG
   //int blockSize[3]={
   //    (int)reg_ceil(controlPointImage->dx / referenceImage->dx),
   //    (int)reg_ceil(controlPointImage->dy / referenceImage->dy),
   //    (int)reg_ceil(controlPointImage->dz / referenceImage->dz),
   //};
   //int sz=referenceImage->nx * referenceImage->ny * referenceImage->nz;
   //int m=referenceImage->nx;
   //int n=referenceImage->ny;
   //int o=referenceImage->nz;
   //int grid_step = blockSize[0];
   //int m1=m/grid_step;
   //int n1=n/grid_step;
   //int o1=o/grid_step;
   //controlPointNumber = m1*n1*o1;
   //DEBUG
   float* message=new float[this->node_number*this->label_nD_num];
   //initialize the energy term with the data cost value
   for(int i=0;i<this->node_number*this->label_nD_num;i++){
      //matrix = discretisedValue (first dimension displacement label, second dim. control point)
      this->regularised_cost[i]=this->discretised_measures[i];
      message[i]=0.0;
   }

   for(int i=0;i<this->label_nD_num;i++){
      cost1[i]=0;
   }

   //weight of the regularisation - constant weight
   //float edgew=this->regularisation_weight + std::numeric_limits<float>::epsilon();
   //float edgew1=1.0f/edgew;

   //calculate mst-cost
   for(int i=(this->node_number-1);i>0;i--){ //do for each control point
      //retreive the child of the current node - start with the leave
      int ochild=this->orderedList[i];//ordered list of all the nodes from root to leaves
      //retreive the parent node of the child
      int oparent=this->parentsList[ochild];
      //retreive the weight of the edge between oparent and ochild
      float edgew=this->edgeWeight[ochild];
      float edgew1=1.0f/edgew;

      for(int l=0;l<this->label_nD_num;l++){
         //matrix = discretisedValue (first dimension displacement label, second dim. control point)
         //weighted by the  edge weight
         cost1[l]=this->regularised_cost[ochild*this->label_nD_num+l]*edgew;
      }

      //fast distance transform
      //It is were the regularisation is calculated
      dt3x(cost1,inds,this->label_1D_num,0,0,0);

      //add mincost to parent node
      for(int l=0;l<this->label_nD_num;l++){
         message[ochild*this->label_nD_num+l]=cost1[l]*edgew1;
         this->regularised_cost[oparent*this->label_nD_num+l]+=cost1[l]*edgew1;
      }
   }

   //backwards pass mst-cost
   for(int i=1;i<this->node_number;i++){ //other direction
      int ochild=this->orderedList[i];
      int oparent=this->parentsList[ochild];
      //retreive the weight of the edge between oparent and ochild
      float edgew=this->edgeWeight[ochild];
      float edgew1=1.0f/edgew;

      for(int l=0;l<this->label_nD_num;l++){
         cost1[l]=(this->regularised_cost[oparent*this->label_nD_num+l]-message[ochild*this->label_nD_num+l]+message[oparent*this->label_nD_num+l])*edgew;
      }

      dt3x(cost1,inds,this->label_1D_num,0,0,0);
      for(int l=0;l<this->label_nD_num;l++){
         message[ochild*this->label_nD_num+l]=cost1[l]*edgew1;
      }

   }

   for(int i=0;i<this->node_number*this->label_nD_num;i++){
      this->regularised_cost[i]+=message[i];
   }

   delete message;
   delete cost1;
   delete vals;
   delete inds;
   reg_print_msg_debug("GetRegularisation_l2 done");
}
/*****************************************************/
/*****************************************************/
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

void dt3x(float* r,int* indr,int rl,float dx,float dy,float dz){
   //rl is length of one side
   for(int i=0;i<rl*rl*rl;i++){
      indr[i]=i;
   }
   //r contains D*(fp) = D(fp)+ Sum(Cc(fp))
   int* v=new int[rl]; //slightly faster if not intitialised in each loop
   float* z=new float[rl+1];
   float* f=new float[rl];
   int* i1=new int[rl];

   //we calculate here the ||up-uq||^2 / ||xp - xq|| ->1st dim => up
   for(int k=0;k<rl;k++){
      for(int i=0;i<rl;i++){
         dt1sq(r+i+k*rl*rl,indr+i+k*rl*rl,rl,-dx,rl,v,z,f,i1);
      }
   }
   //we calculate here the ||up-uq||^2 / ||xp - xq|| ->2nd dim => vp
   for(int k=0;k<rl;k++){
      for(int j=0;j<rl;j++){
         dt1sq(r+j*rl+k*rl*rl,indr+j*rl+k*rl*rl,rl,-dy,1,v,z,f,i1);//);
      }
   }
   //we calculate here the ||up-uq||^2 / ||xp - xq|| ->3rd dim => wp
   for(int j=0;j<rl;j++){
      for(int i=0;i<rl;i++){
         dt1sq(r+i+j*rl,indr+i+j*rl,rl,-dz,rl*rl,v,z,f,i1);//);
      }
   }
   //calculate the min -- of r = Cp(fq) = D(fp)+ Sum(Cc(fp)) + \alpha R(fp,fq)
   float min1=*std::min_element(r,r+rl*rl*rl);
   for(int i=0;i<rl*rl*rl;i++){
      r[i]-=min1;
   }
   delete []i1;
   delete []f;

   delete []v;
   delete []z;
}
