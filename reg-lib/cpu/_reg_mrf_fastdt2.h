#ifndef _REG_MRF_FASTDT2_H
#define _REG_MRF_FASTDT2_H

#include <algorithm>

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

#endif // _REG_MRF_FASTDT2_H
