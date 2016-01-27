#ifndef MATTIAS_H
#define MATTIAS_H
//
#include <algorithm>
/********************************************************************************/
/********************************************************************************/
//Mattias implemention
//boxfilter for patch SSD with certain filter half_width "hw" and temp arrays
void boxfilter(float* input,float* temp1,float* temp2,int hw,int m,int n,int o){

    int sz=m*n*o;
    for(int i=0;i<sz;i++){
        temp1[i]=input[i];
    }

    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=1;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[(i-1)+j*m+k*m*n];
            }
        }
    }

    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<(hw+1);i++){
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n];
            }
            for(int i=(hw+1);i<(m-hw);i++){
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
            }
            for(int i=(m-hw);i<m;i++){
                temp2[i+j*m+k*m*n]=temp1[(m-1)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
            }
        }
    }

    for(int k=0;k<o;k++){
        for(int j=1;j<n;j++){
            for(int i=0;i<m;i++){
                temp2[i+j*m+k*m*n]+=temp2[i+(j-1)*m+k*m*n];
            }
        }
    }

    for(int k=0;k<o;k++){
        for(int i=0;i<m;i++){
            for(int j=0;j<(hw+1);j++){
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n];
            }
            for(int j=(hw+1);j<(n-hw);j++){
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
            for(int j=(n-hw);j<n;j++){
                temp1[i+j*m+k*m*n]=temp2[i+(n-1)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
        }
    }

    for(int k=1;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[i+j*m+(k-1)*m*n];
            }
        }
    }

    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            for(int k=0;k<(hw+1);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n];
            }
            for(int k=(hw+1);k<(o-hw);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
            for(int k=(o-hw);k<o;k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(o-1)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
        }
    }


}

//shift an input image by a certain offset
void imshift(float* input,float* output,int dx,int dy,int dz,int m,int n,int o){
    for(int z=0;z<o;z++){//for every voxel
        for(int y=0;y<n;y++){
            for(int x=0;x<m;x++){
                if(x+dx>=0&&x+dx<m&&y+dy>=0&&y+dy<n&&z+dz>=0&&z+dz<o) //check boundaries
                    output[x+y*m+z*m*n]=input[x+dx+(y+dy)*m+(z+dz)*m*n];
                else
                    output[x+y*m+z*m*n]=input[x+y*m+z*m*n];
            }
        }
    }
}

//input fixed image (with dimensions m x n x o) and half_width of filter
//output (12 x m x n x o) MIND descriptor
void descriptor(float* mind,float* fixed,int m,int n,int o,int filter_hw){

    //MIND with self-similarity context (as in my 2013 MICCAI paper)

    //we first need to evaluate 6 patch distances (SSD) per voxel
    int dx[6]={+filter_hw,+filter_hw,-filter_hw,+0,+filter_hw,+0};
    int dy[6]={+filter_hw,-filter_hw,+0,-filter_hw,+0,+filter_hw};
    int dz[6]={0,+0,+filter_hw,+filter_hw,+filter_hw,+filter_hw};

    //those will be used to fill the 12 descriptor entries
    int sx[12]={-filter_hw,+0,-filter_hw,+0,+0,+filter_hw,+0,+0,+0,-filter_hw,+0,+0};
    int sy[12]={+0,-filter_hw,+0,+filter_hw,+0,+0,+0,+filter_hw,+0,+0,+0,-filter_hw};
    int sz[12]={+0,+0,+0,+0,-filter_hw,+0,-filter_hw,+0,-filter_hw,+0,-filter_hw,+0};
    int index[12]={0,0,1,1,2,2,3,3,4,4,5,5}; //these offsets gather the correct SSDs

    int num_dist=6; //we only have 6 patch SSDs
    int final_length=12; //but a full-length descriptor with 12 entries

    int sz1=m*n*o; //total number of voxels

    //variables to store certain intermediate variables
    float* w1=new float[sz1];
    float* sum1=new float[sz1];
    float* noise1=new float[sz1]; //denominator in MIND-equation
    float* patchSSDs=new float[sz1*num_dist]; //all 6 patch SSDs for every voxel

    for(int i=0;i<sz1*final_length;i++){
        mind[i]=0.0;
    }

    //initialise arrays
    for(int i=0;i<sz1;i++){
        w1[i]=0.0;
        sum1[i]=0.0;
        noise1[i]=0.0;
    }
    for(int i=0;i<sz1*num_dist;i++){
        patchSSDs[i]=0.0;
    }

    //calculate 6 patch SSDs using boxfilter
    float* temp1=new float[sz1]; float* temp2=new float[sz1];
    for(int l=0;l<num_dist;l++){
        imshift(fixed,w1,dx[l],dy[l],dz[l],m,n,o);
        for(int i=0;i<sz1;i++){
            w1[i]=(w1[i]-fixed[i])*(w1[i]-fixed[i]); //point-wise (S)SD
        }
        boxfilter(w1,temp1,temp2,filter_hw,m,n,o);
        for(int i=0;i<sz1;i++){
            patchSSDs[i+l*sz1]=w1[i];
        }
    }
    delete temp1; delete temp2;

    //gather descriptor entries with correct distance offsets
    for(int z=0;z<o;z++){//for every voxel
        for(int y=0;y<n;y++){
            for(int x=0;x<m;x++){
                for(int l=0;l<final_length;l++){
                    if(x+sx[l]>=0&&x+sx[l]<m&&y+sy[l]>=0&&y+sy[l]<n&&z+sz[l]>=0&&z+sz[l]<o){ //check boundaries
                        mind[l+(x+y*m+z*m*n)*final_length]=patchSSDs[(x+sx[l]+(y+sy[l])*m+(z+sz[l])*m*n)+index[l]*sz1];
                    }
                }
            }
        }
    }
    delete patchSSDs;
    //subtract mininum
    for(int i=0;i<sz1;i++){
        float min1=*std::min_element(mind+i*final_length,mind+(i+1)*final_length);
        for(int l=0;l<final_length;l++){
            mind[l+i*final_length]-=min1;
        }
    }
    //calculate noise/variance estimate
    for(int i=0;i<sz1;i++){
        noise1[i]=std::accumulate(mind+i*final_length,mind+(i+1)*final_length,0.0f)/(float)final_length;
    }
    float mean1=std::accumulate(noise1,noise1+sz1,0.0f)/(float)sz1;
    for(int i=0;i<sz1;i++){
        float mean2=std::min(std::max((float)noise1[i],(float)0.001*mean1),(float)1000.0*mean1); //limit variance estimate
        for(int l=0;l<final_length;l++){
            mind[l+i*final_length]=exp(-mind[l+i*final_length]/mean2);

        }
    }
    delete w1;
    delete sum1;
    delete noise1;
}
/********************************************************************************/
/********************************************************************************/
//simply similarity term computation using SAD
void similarityCostSAD(float* dataCost,float* fixed,float* moving,int m,int n,int o,int grid_step,int label_hw,int label_quant,float alpha,int mind_length){

    int label_len=(label_hw*2+1); //length and total size of displacement space
    int label_num=(label_hw*2+1)*(label_hw*2+1)*(label_hw*2+1);
    int m1=m/grid_step; int n1=n/grid_step; int o1=o/grid_step; //dimensions of grid
    int sz1=m1*n1*o1; //number of control points

    //padding of moving image
    int pad1=label_quant*label_hw; int pad2=pad1*2;
    int mp=m+pad2; int np=n+pad2; int op=o+pad2;
    int szp=mp*np*op;
    float* movingpad=new float[szp*mind_length];
    for(int i=0;i<szp*mind_length;i++){
        movingpad[i]=0.0f;
    }
    for(int k=0;k<op;k++){
        for(int j=0;j<np;j++){
            for(int i=0;i<mp;i++){
                //zero-padding
                if(i-pad1>=0&&i-pad1<m&&j-pad1>=0&&j-pad1<n&&k-pad1>=0&&k-pad1<o){
                    for(int l1=0;l1<mind_length;l1++){
                        movingpad[l1+(i+j*mp+k*mp*np)*mind_length]=moving[l1+(i-pad1+(j-pad1)*m+(k-pad1)*m*n)*mind_length];
                    }
                }
            }
        }
    }

    //skip every other voxel within region of each control point for speed (depending on grid spacing)
    int skipz=1; int skipy=1; const int skipx=1;//2;
    /*
    if(grid_step>4){
        skipy=2; skipz=3; //skipx=2;
    }
    if(grid_step>7){
        skipz=3; skipy=3; //skipx=3;
    }
    if(grid_step==4){
        skipz=2; skipz=2;
    }
    */
    //number of sampling points and adapt alpha to it
    //float maxsamp=ceil((float)grid_step/(float)skipx)*ceil((float)grid_step/(float)skipz)*ceil((float)grid_step/(float)skipy);
    //float alphai=(float)grid_step/(alpha*(float)label_quant);
    //float alpha1=0.5*alphai/(float)(maxsamp);
    float alpha1 = 1;

    __m128* movingpad128=(__m128*)movingpad;
    __m128* fixed128=(__m128*)fixed;

#pragma omp parallel for
    for(int z=0;z<o1;z++){ //iterate over all control points
        for(int y=0;y<n1;y++){
            for(int x=0;x<m1;x++){
                //voxel coordinates in fixed image
                int x1=x*grid_step; int y1=y*grid_step; int z1=z*grid_step;

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
                    for(int k=0;k<grid_step;k+=skipz){
                        for(int j=0;j<grid_step;j+=skipy){
                            for(int i=0;i<grid_step;i+=skipx){
                                for(int l1=0;l1<mind_length/4;l1++){
                                    //coordinates in image space
                                    //float t1=fixed[l1+(i+x1+(j+y1)*m+(k+z1)*m*n)*mind_length];
                                    //float t2=movingpad[l1+(i+x2+(j+y2)*mp+(k+z2)*mp*np)*mind_length];
                                    //out1+=fabs(t1-t2); //SAD
                                    __m128 t1=fixed128[l1+(i+x1+(j+y1)*m+(k+z1)*m*n)*mind_length/4];
                                    __m128 t2=movingpad128[l1+(i+x2+(j+y2)*mp+(k+z2)*mp*np)*mind_length/4];
                                    __m128 diff=t1-t2; //SSE speeds-up these distances twofold
                                    dist128+=_mm_max_ps(-diff,diff);
                                }
                            }
                        }
                    }
                    //horizontal sum of SSE array
                    float out4[]={0,0,0,0};
                    _mm_store_ps(out4,dist128);
                    out1=out4[0]+out4[1]+out4[2]+out4[3];

                    //output matrix (first dimension displacement label, second dim. control point)
                    dataCost[(x+y*m1+z*m1*n1)*label_num+l]=out1*alpha1; //control point coordinates
                }
            }
        }
    }
    delete movingpad;
    return;
}
/********************************************************************************/
/********************************************************************************/
#endif // MATTIAS_H
