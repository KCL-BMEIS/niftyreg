
#include "_reg_ReadWriteImage.h"
#include "_reg_ReadWriteBinary.h"
#include "_reg_mrf.h"
#include "_reg_localTrans.h"
#include "_reg_resampling.h"
#include "_reg_mind.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
   time_t start;
   time(&start);

   if(argc!=3) {
      fprintf(stderr, "Usage: %s <indexNeighbours> <edgeWeightMatrix> <expectedOrderedList> <expectedParentsList> <expectedEdgeWeight>\n", argv[0]);
      return EXIT_FAILURE;
   }
   //IO
   char *indexNeighboursName=argv[1];
   char *edgeWeightMatrixName=argv[2];

   //CP
   int nb_CP = 30;
   int nbEdges = nb_CP*6;

   int* indexNeighbours = new int[nbEdges];
   readIntBinaryArray(indexNeighboursName, nbEdges, indexNeighbours);
   //DEBUG
   for(size_t i=0;i<nbEdges;i++) {
       std::cout<<"indexNeighbours[i]="<<indexNeighbours[i]<<std::endl;
   }
   //DEBUG
   //
   float* edgeWeightMatrix = new float[nbEdges];
   readFloatBinaryArray(edgeWeightMatrixName, nbEdges, edgeWeightMatrix);
   //DEBUG
   for(size_t i=0;i<nbEdges;i++) {
       std::cout<<"edgeWeightMatrix[i]="<<edgeWeightMatrix[i]<<std::endl;
   }
   std::cout<<"DATA READ"<<std::endl;
   //DEBUG
   //
   reg_mrf* reg_mrfObject =
           new reg_mrf(2,1,0,3,nb_CP);
   std::cout<<"OBJECT DONE"<<std::endl;

   reg_mrfObject->GetPrimsMST(edgeWeightMatrix,indexNeighbours, nb_CP, 6, false);
   //PRINT THE RESULTS
   int* olP = reg_mrfObject->GetOrderedListPtr();
   int* plP = reg_mrfObject->GetParentsListPtr();
   float* ewP = reg_mrfObject->GetEdgeWeightPtr();
   //
   for(size_t i=0;i<nb_CP;i++) {
       std::cout<<i+1<<std::endl;
       std::cout<<"olP[i]="<<olP[i]+1<<std::endl;
       std::cout<<"plP[i]="<<plP[i]+1<<std::endl;
       std::cout<<"ewP[i]="<<ewP[i]<<std::endl;
   }
   //
   delete[] indexNeighbours;
   delete[] edgeWeightMatrix;
   //
#ifndef NDEBUG
   printf("All good\n");
#endif
   return EXIT_SUCCESS;
}
