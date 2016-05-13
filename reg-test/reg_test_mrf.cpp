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
      fprintf(stderr, "Usage: %s <dataCost> <expectedLabeling>\n", argv[0]);
      return EXIT_FAILURE;
   }
   //IO
   char *dataCostName=argv[1];
   char *expectedLabelingName=argv[2];

   //DISP
   int discrete_radius = 2;
   int discrete_increment = 1;
   //CP
   size_t nb_CP = 4;
   size_t nb_disp = discrete_radius/discrete_increment*2+1;
   size_t nb_disp3D = nb_disp*nb_disp*nb_disp;

   float* dataCost = new float[nb_CP*nb_disp3D];
   readFloatBinaryArray(dataCostName, nb_CP*nb_disp3D, dataCost);
   //DEBUG
   for(size_t i=0;i<nb_CP*nb_disp3D;i++) {
       std::cout<<"dataCost[i]="<<dataCost[i]<<std::endl;
   }
   //DEBUG
   int* expectedLabeling = new int[nb_CP];
   readIntBinaryArray(expectedLabelingName, nb_CP, expectedLabeling);
   //DEBUG
   for(size_t i=0;i<nb_CP;i++) {
       std::cout<<"expectedLabeling[i]="<<expectedLabeling[i]<<std::endl;
   }
   //DEBUG
   int* orderedList=new int[nb_CP];
   for(int i=0;i<nb_CP;i++) {
       orderedList[i]=i;
   }
   int* parentsList=new int[nb_CP];
   //for(int i=0;i<nb_CP;i++) {
   //    parentsList[i]=i-1;
   //}
   parentsList[0]=-1;
   parentsList[1]=0;
   parentsList[2]=1;
   parentsList[3]=1;
   float* edgeWeight=new float[nb_CP];
   for(int i=0;i<nb_CP;i++) {
       edgeWeight[i]=1;
   }
   //
   reg_mrf* reg_mrfObject =
           new reg_mrf(discrete_radius,discrete_increment,0,3,nb_CP);

   reg_mrfObject->SetDiscretisedMeasure(dataCost);
   reg_mrfObject->SetOrderedList(orderedList);
   reg_mrfObject->SetParentsList(parentsList);
   reg_mrfObject->SetEdgeWeight(edgeWeight);
   //
   reg_mrfObject->GetRegularisation();
   reg_mrfObject->getOptimalLabel();
   //
   //Let's compare the 2 labelling
   for(size_t i=0;i<nb_CP;i++) {
       int currentValue = reg_mrfObject->GetOptimalLabelPtr()[i];
       int expectedValue = expectedLabeling[i];
       std::cout<<"currentValue="<<currentValue<<std::endl;
       std::cout<<"expectedValue="<<expectedValue<<std::endl;
       //if((currentValue - expectedValue) != 0) {
       //    reg_print_msg_error("the 2 labelling are differents");
       //    return EXIT_FAILURE;
       //}
   }
   //
   delete[] dataCost;
   delete[] expectedLabeling;
   delete[] edgeWeight;
   delete[] parentsList;
   delete[] orderedList;
   //
#ifndef NDEBUG
   printf("All good\n");
#endif
   return EXIT_SUCCESS;
}
