 /*
 *  reg_ppcnr.cpp
 *
 *
 *  Created by Andrew Melbourne on 16/11/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *  (Last modified 23/01/2013)
 *
 */

#include "_reg_tools.h"
#include "float.h"
#include <limits>
#include <string.h>

#ifdef _WIN32
    #include <time.h>
#endif

#define PrecisionTYPE float

typedef struct{
	char *sourceImageName;
	char *affineMatrixName;
	char *inputCPPName;
    char *targetMaskName;
	const char *outputImageName;
	char *currentImageName;
	float spacing[3];
	int maxIteration;
	int prinComp;
    int locality;
    int singletimepoint;
	const char *outputResultName;
	char *outputCPPName;
}PARAM;

typedef struct{
	bool sourceImageFlag;
	bool affineMatrixFlag;
	bool affineFlirtFlag;
	bool prinCompFlag;
    bool localityFlag;
    bool singleFlag;
	bool meanonly;
	bool outputResultFlag;
	bool outputCPPFlag;
	bool backgroundIndexFlag;
	bool pca0;
    bool pca1;
	bool pca2;
    bool pca3;
    bool pca4;
	bool aladin;
	bool flirt;
    bool autolevel;
    bool makesourcex;
}FLAG;


void PetitUsage(char *exec)
{
	fprintf(stderr,"PROGRESSIVE PRINCIPAL COMPONENT REGISTRATION (PPCNR).\n");
	fprintf(stderr,"Fast Free-Form Deformation algorithm for dynamic contrast enhanced (DCE) non-rigid registration.\n");
	fprintf(stderr,"Usage:\t%s -source <sourceImageName> [OPTIONS].\n",exec);
	fprintf(stderr,"\t\t\t\t*Note that no target image is needed!\n");
	fprintf(stderr,"\tSee the help for more details (-h).\n");
	return;
}
void Usage(char *exec)
{
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("PROGRESSIVE PRINCIPAL COMPONENT REGISTRATION (PPCNR).\n");
	printf("Fast Free-Form Deformation algorithm for non-rigid DCE-MRI registration.\n");
    printf("This implementation is a re-factoring of the PPCR algorithm in:\n");
	printf("Melbourne et al., \"Registration of dynamic contrast-enhanced MRI using a \n");
	printf(" progressive principal component registration (PPCR)\", Phys Med Biol, 2007.\n");		
	printf("This code has been written by Andrew Melbourne (a.melbourne@cs.ucl.ac.uk)\n");
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Usage:\t%s -source <filename> [OPTIONS].\n",exec);
	printf("\t-source <filename>\tFilename of the source image (mandatory)\n");
    printf("     Or -makesource <n> <filenames> \tThis will generate a 4D volume from the n filenames (saved to 'source4D.nii'.\n");
	printf("\t*Note that no target image is needed!\n");
    printf("\n*** Main options:\n");
    printf("\t-result <filename> \tFilename of the resampled image [ppcnrfinal.nii].\n");
    printf("\t-cpp <filename>\t\tFilename of final 5D control point grid (non-rigid registration only).\n");
    printf("     Or -aff <filename>\t\tFilename of final concatenated affine transformation (affine registration only).\n");
	printf("\t-prinComp <int>\t\tNumber of principal component iterations to run [#timepoints/2].\n");
    printf("\t-maxit <int>\t\tNumber of registration iterations to run [max(400/prinComp,50)].\n");
    printf("\t-autolevel \t\tAutomatically increase registration level during PPCR (switched off with -ln or -lp options).\n"); // not with -FLIRT
    printf("\t-pca0 <filename> \tOutput pca images 1:prinComp without registration step [pcaX.nii].\n"); // i.e. just print out each PCA image.
    printf("\t-pca1 <filename> \tOutput pca target images 1:prinComp for inspection during registration [pcaX.nii].\n");
    printf("\t-pca2 <filename> \tOutput intermediate results 1:prinComp for inspection [outX.nii].\n");
    printf("\t-pca3 <filename> \tOutput intermediate transformations 1:prinComp for inspection [dofX.nii/dofX.txt].\n");
    printf("\t-pca4 <filename> \tSave current results only.\n"); // i.e. removes previous outputs...
    printf("\n*** Registration to mean options (No PPCR):\n");
    printf("\t-mean \t\t\tIterative registration to the mean image only.\n"); // registration to the mean is quite inefficient as it uses the ppcr 4D->4D model.
    printf("\t-locality <int>\t\tNumber of images to form local mean (#images either side) [#timepoints].\n");
    printf("\t-single <int>\t\tRegister to a single timepoint (1 is first image).\n");
    printf("\t-prinComp <int>\t\tSets the number of iterations to the mean to run [#timepoints/2].\n");    
    //printf("\t-flirt \t\t\tfor PPCNR using Flirt affine registration (not tested)\n");
    printf("\n*** reg_f3d/reg_aladin options are carried through (use reg_f3d -h or reg_aladin -h to see these options).\n");
    //system("reg_f3d -h");
    
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	return;
}


int main(int argc, char **argv)
{
	time_t start; time(&start);
	
	PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
	FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
	flag->aladin=0;
	flag->flirt=0;
	flag->pca0=0;
    flag->pca1=0;
    flag->pca2=0;
    flag->pca3=0;
    flag->pca4=0;
	flag->meanonly=0;
    flag->autolevel=0;
	flag->outputCPPFlag=0;
	flag->outputResultFlag=0;
    flag->makesourcex=0;
    flag->localityFlag=0;
    flag->singleFlag=0;
    param->maxIteration=-1;
    param->singletimepoint=1;

	char regCommandAll[1055]="";
	char regCommand[1000]="";strcat(regCommand,"-target anchorx.nii -source floatx.nii");
	char regCommandF[1000]="";strcat(regCommandF,"flirt -ref anchorx.nii -in floatx.nii -out outputResult.nii.gz");
	char style[10]="";
	char STYL3[10]="";
	
	/* read the input parameters */
	for(int i=1;i<argc;i++){
		if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
			strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
			strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
			Usage(argv[0]);
			return 0;
		}
		else if(strcmp(argv[i], "-source") == 0){
			param->sourceImageName=argv[++i];
			flag->sourceImageFlag=1;
        }
        else if(strcmp(argv[i], "-makesource") == 0 || strcmp(argv[i], "-makesourcex")==0){
            if(strcmp(argv[i], "-makesourcex")==0){flag->makesourcex=1;}
            nifti_image *source = nifti_image_read(argv[i+2],false);
            nifti_image *makesource = nifti_copy_nim_info(source);
            nifti_image_free(source);
            makesource->ndim=makesource->dim[0] = 4;
            makesource->nt = makesource->dim[4] = atoi(argv[++i]);
            makesource->nvox =(size_t)makesource->nx*(size_t)makesource->nz*(size_t)makesource->ny*(size_t)makesource->nt;
            makesource->data = (void *)malloc(makesource->nvox * makesource->nbyper);
			char *temp_data = reinterpret_cast<char *>(makesource->data);
            for(int ii=0;ii<makesource->nt;ii++){ // fill with file data
                source = nifti_image_read(argv[++i],true);				
                memcpy(&(temp_data[ii*source->nvox*source->nbyper]), source->data, source->nbyper*source->nvox);
                nifti_image_free(source); 
            }            
            nifti_set_filenames(makesource,"source4D.nii", 0, 0); // might want to set this 
            nifti_image_write(makesource);
            nifti_image_free(makesource);
            char buffer[20];
            sprintf(buffer,"source4D.nii");
            param->sourceImageName=buffer;
            flag->sourceImageFlag=1;  
        }
		else if(strcmp(argv[i], "-target") == 0){
		    printf("Target image is not necessary!");
			PetitUsage(argv[0]);
		}
		else if(strcmp(argv[i], "-aff") == 0){ // use ppcnr affine
			param->outputCPPName=argv[++i];
			flag->outputCPPFlag=1;
            flag->aladin=1;
		}
        else if(strcmp(argv[i], "-incpp") == 0){ // remove -incpp option
			printf("-incpp will not be used!");
        }
		else if(strcmp(argv[i], "-result") == 0){
			param->outputResultName=argv[++i];
			flag->outputResultFlag=1;
		}
        else if(strcmp(argv[i], "-cpp") == 0){
			param->outputCPPName=argv[++i];
			flag->outputCPPFlag=1;
		}
		else if(strcmp(argv[i], "-prinComp") == 0){ // number of pcs to use
			param->prinComp=atoi(argv[++i]);
			flag->prinCompFlag=1;
		}
        else if(strcmp(argv[i], "-locality") == 0){ // number of local images to use
			param->locality=atoi(argv[++i]);
			flag->localityFlag=1;
		}
        else if(strcmp(argv[i], "-single") == 0){ // number of local images to use
			param->singletimepoint=atoi(argv[++i]);
			flag->singleFlag=1;
		} 
		else if(strcmp(argv[i], "-pca0") == 0){ // write pca images during registration
			flag->pca0=1;
		}
        else if(strcmp(argv[i], "-pca1") == 0){ // write pca images without registration
			flag->pca1=1;
		}
        else if(strcmp(argv[i], "-pca2") == 0){ // write output images during registration
			flag->pca2=1;
		}
        else if(strcmp(argv[i], "-pca3") == 0){ // write output images during registration
			flag->pca3=1;
		}
        else if(strcmp(argv[i], "-pca4") == 0){ // write output images during registration
			flag->pca4=1;
		}
        else if(strcmp(argv[i], "-pca123") == 0){ // combined
			flag->pca1=1;
            flag->pca2=1;
            flag->pca3=1;
		}
        else if(strcmp(argv[i], "-mean") == 0){ // iterative registration to the mean
			flag->meanonly=1;
		}
		else if(strcmp(argv[i], "-flirt") == 0){ // one day there will be a flirt option:)
			flag->flirt=1;
		}
        else if(strcmp(argv[i], "-autolevel") == 0){
			flag->autolevel=1;
		}
        else if(strcmp(argv[i], "-lp") == 0){  // force autolevel select off if lp or ln are present.
            flag->autolevel=0;
            strcat(regCommand," ");
			strcat(regCommand,argv[i]);
            strcat(regCommand," ");
			strcat(regCommand,argv[i+1]);
            ++i;
        }
        else if(strcmp(argv[i], "-ln") == 0){  // force autolevel select off if lp or ln are present.
            flag->autolevel=0;
            strcat(regCommand," ");
			strcat(regCommand,argv[i]);
            strcat(regCommand," ");
			strcat(regCommand,argv[i+1]);
            ++i;
        }
        else if(strcmp(argv[i], "-maxit") == 0){ // extract number of registration iterations for display
            param->maxIteration=atoi(argv[i]);
            strcat(regCommand," ");
			strcat(regCommand,argv[i]);
            strcat(regCommand," ");
			strcat(regCommand,argv[i+1]);
            ++i;
        }
        else{
            strcat(regCommand," ");
			strcat(regCommand,argv[i]);
        }
	}
    if(flag->makesourcex){return 0;} // stop if being used to concatenate 3D images into 4D object.
    
	if(!flag->sourceImageFlag){
		fprintf(stderr,"Error:\tAt least define a source image!\n");
		Usage(argv[0]);
		return 1;
	}
	
	nifti_image *image = nifti_image_read(param->sourceImageName,true);
	if(image == NULL){
		fprintf(stderr,"* ERROR Error when reading image: %s\n",param->sourceImageName);
		return 1;
	}
    reg_tools_changeDatatype<PrecisionTYPE>(image); // FIX DATA TYPE - DOES THIS WORK?
    
	if(!flag->prinCompFlag) param->prinComp=(int)(image->nt/2);// Check the number of components
	if(param->prinComp>=image->nt) param->prinComp=image->nt-1;
    if(!flag->localityFlag) param->locality=(int)(image->nt);
	if(flag->singleFlag) param->prinComp=1; // limit number of iterations!
    if(!flag->outputResultFlag) param->outputResultName="ppcnrfinal.nii";
	if(param->maxIteration<0) param->maxIteration=(int)(400/param->prinComp); // number of registraton iterations is automatically set here...
    param->maxIteration=(param->maxIteration<50)?50:param->maxIteration;
    if(!flag->outputCPPFlag){
		char buffer[40];
        sprintf(buffer,"ppcnrfinal-%s.nii",STYL3);
		param->outputCPPName=buffer;
	}
	if(flag->aladin){ // decide whether to use affine or free-form
		strcat(regCommandAll,"reg_aladin ");
		strcat(style,"aff");
		strcat(STYL3,"AFF");
	}
	else if(flag->flirt){
		strcat(style,"aff");
	}
	else{
		strcat(regCommandAll,"reg_f3d ");
		strcat(style,"cpp");
		strcat(STYL3,"CPP");
	}	
	strcat(regCommandAll,regCommand);
    printf("%s\n",style);
		
	/* ****************** */
	/* DISPLAY THE REGISTRATION PARAMETERS */
	/* ****************** */
    
	printf("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	printf("Command line:\n %s",argv[0]);
	for(int i=1;i<argc;i++)
		printf(" %s",argv[i]);
	printf("\n\n");
	if(flag->meanonly){
		printf("Iterative Registration to the Mean Only (Algorithm will ignore PCA results)----------------\n"); // quite an inefficient method, but it saves me writing the code from scratch:) seems to work too...
	}
	else{
		printf("PPCNR Parameters\n----------------\n");
	}
	printf("Source image name: %s\n",param->sourceImageName);
	printf("Number of timepoints: %i \n", image->nt);
	printf("Number of principal components: %i\n",param->prinComp);
	printf("Registration max iterations: %i\n",param->maxIteration);
	
	/* ********************** */
	/* START THE REGISTRATION */
	/* ********************** */
	param->outputImageName="anchor.nii";   // NEED TO GET WORKING AND PUT INTERMEDIATE FILES IN SOURCE DIRECTORY.
	nifti_image *images=nifti_copy_nim_info(image); // Need to make a new image that has the same info as the original.
	images->data = (PrecisionTYPE *)calloc(images->nvox, image->nbyper);
	memcpy(images->data, image->data, image->nvox*image->nbyper);
	
	/* ************************************/
	/* FOR NUMBER OF PRINCIPAL COMPONENTS */
	/* ************************************/   
    
	float levels[3];
	float *vsum = new float [param->prinComp];
	for(int i=0;i<param->prinComp;i++) vsum[i]=0.f;
	float *dall = new float [images->nt*param->prinComp];
	levels[0]=-10.0;levels[1]=-5.0;levels[2]=-2.5;
	int levelNumber=1;if(images->nt<3) levelNumber=3;
	PrecisionTYPE *Mean = new PrecisionTYPE [image->nt];
    PrecisionTYPE *Cov = new PrecisionTYPE [image->nt*image->nt];
    PrecisionTYPE cov;
    char pcaname[20]; 
    char outname[20];
    char dofname[20];
    
    for(int prinCompNumber=1;prinCompNumber<=param->prinComp;prinCompNumber++)
	{
		param->spacing[0]=levels[(int)(3.0*prinCompNumber/(param->prinComp+1))]; // choose a reducing level number	
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
		printf("RUNNING ITERATION %i of %i \n",prinCompNumber, param->prinComp);
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
		printf("Running component %i of %i \n", prinCompNumber, param->prinComp);
		if(flag->autolevel){printf("Running %i levels at %g spacing \n", levelNumber, param->spacing[0]);}
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
		
		// Read images and find image means
        size_t voxelNumber = image->nvox/image->nt;
		PrecisionTYPE *intensityPtr = static_cast<PrecisionTYPE *>(image->data);
		for(int t=0;t<image->nt;t++){
			Mean[t]=0.f;
            for(size_t i=0;i<voxelNumber; i++){
				Mean[t] += *intensityPtr++;
			}
			Mean[t] /= (PrecisionTYPE)voxelNumber;
		}
	
		// calculate covariance matrix
		intensityPtr = static_cast<PrecisionTYPE *>(image->data);
		for(int t=0;t<image->nt;t++){
			PrecisionTYPE *currentIntensityPtr2 = &intensityPtr[t*voxelNumber];
			for(int t2=t;t2<image->nt;t2++){
				PrecisionTYPE *currentIntensityPtr1 = &intensityPtr[t*voxelNumber];
				cov=0.f;
                for(size_t i=0;i<voxelNumber; i++){
					cov += (*currentIntensityPtr1++ - Mean[t]) * (*currentIntensityPtr2++ - Mean[t2]);
				}
				Cov[t+image->nt*t2]=cov/(PrecisionTYPE)voxelNumber;
				Cov[t2+image->nt*t]=Cov[t+image->nt*t2]; // covariance matrix is symmetric.
			}
		}

		// calculate eigenvalues/vectors, courtesy of numerical recipes.
		// 1. reduce
		int n=image->nt;
		float EPS=1e-15;
		int l,k,j,i;
		float scale,hh,h,g,f;
		float *d = new float [n];
        float *e = new float [n];
        float *z = new float [n*n];
        for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				z[i+n*j]=Cov[i+n*j];
			}
		}
		for (i=n-1;i>0;i--){
			l=i-1;
			h=scale=0.0;
			if(l>0){
				for(k=0;k<i;k++)
					scale+=abs(z[i+n*k]);
				if (scale==0.0)
					e[i]=z[i+n*l];
				else {
					for(k=0;k<i;k++){
						z[i+n*k] /= scale;
						h+=z[i+n*k]*z[i+n*k];
					}
					f=z[i+n*l];
					g=(f>=0.0 ? -sqrt(h) : sqrt(h));
					e[i]=scale*g;
					h-=f*g;
					z[i+n*l]=f-g;
					f=0.0;
					for (j=0;j<i;j++){
						z[j+n*i]=z[i+n*j]/h;
						g=0.0;
						for (k=0;k<j+1;k++)
							g+=z[j+n*k]*z[i+n*k];
						for (k=j+1;k<i;k++)
							g+= z[k+n*j]*z[i+n*k];
						e[j]=g/h;
						f+=e[j]*z[i+n*j];
					}
					hh=f/(h+h);
					for (j=0;j<i;j++){
						f=z[i+n*j];
						e[j]=g=e[j]-hh*f;
						for (k=0;k<j+1;k++)
							z[j+n*k]-=(f*e[k]+g*z[i+n*k]);
					}
				}
			} else
				e[i]=z[i+n*l];
			d[i]=h;
		}
		d[0]=0.0;
		e[0]=0.0;
		for (i=0;i<n;i++){
			if(d[i]!=0.0){
				for (j=0;j<i;j++){
					g=0.0;
					for (k=0;k<i;k++)
						g+=z[i+n*k]*z[k+n*j];
					for (k=0;k<i;k++)
						z[k+n*j]-=g*z[k+n*i];
				}
			}
			d[i]=z[i+n*i];
			z[i+n*i]=1.0;
			for (j=0;j<i;j++) z[j+n*i]=z[i+n*j]=0.0;
		}
		
		printf("Image Means=[%g",Mean[0]);
		for(int i=1;i<image->nt;i++){
			printf(",%g",Mean[i]); // not sure it's quite right...
		}
		printf("]\n");
		for(int i=0;i<image->nt;i++){
			printf("Cov=[%g",Cov[i+n*0]);
			for(int j=1;j<image->nt;j++){
				printf(",%g",Cov[i+n*j]);
			}
			printf("]\n");
		}

		// 2. diagonalise
		int m,iter;
		float s,r,p,dd,c,b;
		for (i=1;i<n;i++) e[i-1]=e[i];
		e[n-1]=0.0;
		for (l=0;l<n;l++){
			iter=0;
			do {
				for (m=l;m<n-1;m++) {
					dd=abs(d[m])+abs(d[m+1]);
					if(abs(e[m])<=EPS*dd) break;
				}
				if(m!=l){
					if(iter++==30) break;
					g=(d[l+1]-d[l])/(2.0*e[l]);
					r=sqrt(g*g+1.0);
					g=d[m]-d[l]+e[l]/(g+abs(r)*g/abs(g));
					s=c=1.0;
					p=0.0;
					for (i=m-1;i>=l;i--){
						f=s*e[i];
						b=c*e[i];
						e[i+1]=(r=sqrt(f*f+g*g));
						if(r<EPS){
							d[i+1]-=p;
							e[m]=0.0;
							break;
						}
						s=f/r;
						c=g/r;
						g=d[i+1]-p;
						r=(d[i]-g)*s+2.0*c*b;
						d[i+1]=g+(p=s*r);
						g=c*r-b;
						for (k=0;k<n;k++){
							f=z[k+n*(i+1)];
							z[k+n*(i+1)]=s*z[k+n*i]+c*f;
							z[k+n*i]=c*z[k+n*i]-s*f;
						}
					}
					if(r<EPS && i>=l) continue;
					d[l]-=p;
					e[l]=g;
					e[m]=0.0;
				}
				// printf("Iterations=%i\n",iter);
			} while(m!=l);
		} // Seems to be ok for an arbitrary covariance matrix.	
		
		// 3. sort eigenvalues & eigenvectors
		for(int i=0;i<n-1;i++){
			float p=d[k=i];
			for(int j=i;j<n;j++)
				if(d[j]>=p) p=d[k=j];
			if(k!=i){
				d[k]=d[i];
				d[i]=p;
				if(z != NULL)
					for(int j=0;j<n;j++){
						p=z[j+n*i];
						z[j+n*i]=z[j+n*k];
						z[j+n*k]=p;
					}
			}
		}
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
		for(int i=0;i<image->nt;i++){
			printf("EVMatrix=[%g",z[i+n*0]);
			for(int j=1;j<image->nt;j++){
				printf(",%g",z[i+image->nt*j]);
			}
			printf("]\n");
		}
		printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
		printf("Eigenvalues=[%g",d[0]);
		for(int i=0;i<image->nt;i++){
			if(i>0){
				printf(",%g",d[i]);
			}
			vsum[prinCompNumber-1]+=d[i];
			dall[i+image->nt*(prinCompNumber-1)]=d[i];
		}
		printf("]\n");
		for(j=0;j<prinCompNumber;j++){
			printf("Variances(%i)=[%g",j+1,100.0*dall[0+n*j]/vsum[j]);
			for(int i=1;i<image->nt;i++){
				printf(",%g",100.0*dall[i+image->nt*j]/vsum[j]);
			}
			printf("]\n");
		}
		if(flag->meanonly){
			printf("Iterative registration to mean only - eigenvector matrix overwritten.\n");
			for(int i=0;i<image->nt;i++){
				for(int j=0;j<image->nt;j++){
					z[i+image->nt*j]=1.0/sqrtf(image->nt*prinCompNumber); // is this right?! - if using NMI it's rather moot so I'm not too bothered at the moment...
				}
			}
		}
		// 4. rebuild images
		nifti_image *imagep=nifti_copy_nim_info(image); // Need to make a new image that has the same info as the original.
		imagep->data = (PrecisionTYPE *)calloc(imagep->nvox, image->nbyper);
		int bottom,top;
        float dotty,norman;
		PrecisionTYPE *intensityPtr1 = static_cast<PrecisionTYPE *>(image->data);
        PrecisionTYPE *intensityPtr2 = static_cast<PrecisionTYPE *>(imagep->data);
        for(size_t i=0;i<voxelNumber; i++){
            if(flag->singleFlag){
                for(int t=0;t<image->nt;t++){ // 4.3) Copy the correct image to the imagep data
                    intensityPtr2[t*voxelNumber+i] = intensityPtr1[(param->singletimepoint-1)*voxelNumber+i];			
                }         
            }
            else{
                if(!flag->localityFlag){
                    for(int c=0;c<prinCompNumber;c++){ // Add up component contributions
                        dotty=0.0;
                        for(int t=0;t<image->nt;t++){ // 4.1.1) Multiply each element by eigenvector and add (I.e. dot product)
                            dotty += intensityPtr1[t*voxelNumber+i] * z[t+image->nt*c];	
                        }
                        for(int t=0;t<image->nt;t++){ // 4.1.2) Multiply value above by that eigenvector and write these to the imagep data
                            intensityPtr2[t*voxelNumber+i] += dotty * z[t+image->nt*c];			
                        }
                    }
                }
                else{                
                    for(int t=0;t<image->nt;t++){ // 4.2) Find local mean and write to the imagep data
                        dotty=0.0;
                        norman=0.0;
                        bottom=(0<t-param->locality)?t-param->locality:0;
                        top=!(t+param->locality>image->nt)?t+param->locality:image->nt;
                        for(int tt=bottom;tt<=top;tt++){ 
                            dotty += intensityPtr1[tt*voxelNumber+i];
                            norman += 1.0;
                        }
                        intensityPtr2[t*voxelNumber+i] += dotty/norman;			
                    }
                }              
            }
        }
        if(flag->pca4 & (prinCompNumber>2)){
            if(flag->pca0 | flag->pca1){remove(pcaname);} // easy to code, but this deletes previous output before writing the next which is a little risky...
            if(flag->pca2){remove(outname);}
            if(flag->pca3){remove(dofname);}
        }
		char pcaname[20];        
        sprintf(pcaname,"pca%i.nii",prinCompNumber);
        nifti_set_filenames(imagep,pcaname, 0, 0);
        if(flag->pca0 | flag->pca1){nifti_image_write(imagep);}
	
		if(!flag->pca0){
            /* ****************************/										
            /* FOR NUMBER OF 'TIMEPOINTS' */
            /* ****************************/
            // current: images // these are both open: perpetual source
            // target:  imagep //					   pca target
            PrecisionTYPE *intensityPtrP = static_cast<PrecisionTYPE *>(imagep->data); // pointer to pca-anchor data
            PrecisionTYPE *intensityPtrS = static_cast<PrecisionTYPE *>(images->data); // pointer to real source-float data
            PrecisionTYPE *intensityPtrC = static_cast<PrecisionTYPE *>(image->data); // pointer to updated 'current' data
            for(int imageNumber=0;imageNumber<images->nt;imageNumber++)
            {		
                // ROLLING FLOAT AND ANCHOR IMAGES
                nifti_image *stores = nifti_copy_nim_info(images);
                stores->ndim=stores->dim[0]=3;
                stores->nt=stores->dim[4]=1;
                stores->nvox =(size_t)stores->nx*(size_t)stores->ny*(size_t)stores->nz;
                stores->data = (void *)calloc(stores->nvox,images->nbyper);

                nifti_image *storet = nifti_copy_nim_info(stores);
                storet->data = (void *)calloc(storet->nvox, storet->nbyper);
                    
                // COPY THE APPROPRIATE VALUES
                PrecisionTYPE *intensityPtrPP = static_cast<PrecisionTYPE *>(storet->data); // 3D real source image (needs current cpp image)
                PrecisionTYPE *intensityPtrSS = static_cast<PrecisionTYPE *>(stores->data); // 3D pca-float data
                memcpy(intensityPtrPP, &intensityPtrP[imageNumber*storet->nvox], storet->nvox*storet->nbyper);
                memcpy(intensityPtrSS, &intensityPtrS[imageNumber*stores->nvox], stores->nvox*stores->nbyper);

                nifti_set_filenames(stores,"outputResult.nii", 0, 0); // Fail-safe for reg_f3d failure
                nifti_image_write(stores);
                nifti_set_filenames(stores,"floatx.nii", 0, 0); // TODO NAME
                nifti_image_write(stores);
                nifti_image_free(stores);
                nifti_set_filenames(storet,"anchorx.nii", 0, 0); // TODO NAME
                nifti_image_write(storet);
                nifti_image_free(storet);			
                
                char regCommandB[1055]="";
                if(!flag->flirt){				
                    sprintf(regCommandB,"%s -%s ",regCommandAll,style);
                    char buffer[20];
                    if(flag->aladin){
                        sprintf(buffer,"float%s%i.txt", style,imageNumber+1);
                    }
                    else{
                        sprintf(buffer,"float%s%i.nii", style,imageNumber+1);
                    }
                    strcat(regCommandB,buffer);
                    char buffer2[30];
                    if(flag->autolevel){
                        sprintf(buffer2," -ln %i",levelNumber);
                        strcat(regCommandB,buffer2);
                        char buffer3[20];
                        if(!flag->aladin)
                            sprintf(buffer3," -sx %g",param->spacing[0]);
                        strcat(regCommandB,buffer3);
                    }
                    if(prinCompNumber>1){
                        char buffer4[8];
                        sprintf(buffer4," -in%s ",style);
                        strcat(regCommandB,buffer4);
                        strcat(regCommandB,buffer);
                    }
                }
                else{ // flirt -ref -in -out -omat -init
                    sprintf(regCommandB,"%s -omat ",regCommandF);
                    char buffer[20];
                    sprintf(buffer,"float%s%i.txt", style,imageNumber+1);
                    strcat(regCommandB,buffer);
                    if(prinCompNumber>1){
                        char buffer3[8];
                        sprintf(buffer3," -init ");
                        strcat(regCommandB,buffer3);
                        strcat(regCommandB,buffer);
                        strcat(regCommandB,";gunzip -f outputResult.nii.gz");
                    }
                }
                
                // DO REGISTRATION
                printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
                printf("RUNNING ITERATION %i of %i \n",prinCompNumber, param->prinComp);
                printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
                printf("Registering image %i of %i \n", imageNumber+1,images->nt);
                printf("'%s' \n",regCommandB);
				if(system(regCommandB)){
					fprintf(stderr, "Error while running the following command:\n%s\n",regCommandB);
					exit(EXIT_FAILURE);
				}
                
                // READ IN RESULT AND MAKE A NEW CURRENT IMAGE 'image'			
                stores = nifti_image_read("outputResult.nii",true); // TODO NAME
                PrecisionTYPE *intensityPtrCC = static_cast<PrecisionTYPE *>(stores->data); // 3D result image
                memcpy(&intensityPtrC[imageNumber*stores->nvox], intensityPtrCC, stores->nvox*stores->nbyper);	
                nifti_image_free(stores);
            }
        }
		nifti_image_free(imagep);
        char outname[20];
        sprintf(outname,"out%i.nii",prinCompNumber);
        nifti_set_filenames(image,outname, 0, 0);
        if(!flag->pca0 & flag->pca2){nifti_image_write(image);}
        if(!flag->pca0 & flag->pca3){
            char dofname[20];
            if(!flag->aladin & !flag->flirt){
                sprintf(dofname,"dof%i.nii",prinCompNumber);
                char buffer[20];
                sprintf(buffer,"float%s1.nii",style);
                nifti_image *dof = nifti_image_read(buffer,true);
                nifti_image *dofs = nifti_copy_nim_info(dof);
                dofs->nt = dofs->dim[4] = images->nt;
                dofs->nvox = dof->nvox*images->nt;
                dofs->data = (PrecisionTYPE *)calloc(dofs->nvox, dof->nbyper);
                PrecisionTYPE *intensityPtrD = static_cast<PrecisionTYPE *>(dofs->data);
                for(int t=0;t<images->nt;t++){
                    char buffer[20];
                    sprintf(buffer,"float%s%i.nii",style, t+1);
                    nifti_image *dof = nifti_image_read(buffer,true);
                    PrecisionTYPE *intensityPtrDD = static_cast<PrecisionTYPE *>(dof->data);
                    int r=dof->nvox/3.0;
                    for(int i=0;i<3;i++){
                        memcpy(&intensityPtrD[i*image->nt*r+t*r], &intensityPtrDD[i*r], dof->nbyper*r);	
                    }		
                    nifti_image_free(dof);
                }
                nifti_set_filenames(dofs,dofname, 0, 0); // TODO NAME 	// write final dof data
                nifti_image_write(dofs);
                nifti_image_free(dofs);
            }
            else{
                sprintf(dofname,"dof%i.txt",prinCompNumber);
                std::string final_string = "";
                for(int t=0;t<images->nt;t++){
                    char buffer[20];
                    sprintf(buffer,"float%s%i.txt",style,t+1);
                    std::ifstream ifs(buffer);  
                    std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    final_string+=str;               
                }
                std::ofstream ofs(dofname);
                ofs<<final_string.c_str();
            }            
        }        
        //if(flag->singleFlag) break; // only run one iteration for registration to single timepoint.
    } // End PC's
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Finished Iterations and now writing outputs...\n");
    
	// WRITE OUT RESULT IMAGE AND RESULT DOF
	// Read in images and put into single object
	if(!flag->pca0){
        if(!flag->aladin & !flag->flirt){
            char buffer[20];
            sprintf(buffer,"float%s1.nii",style);
            nifti_image *dof = nifti_image_read(buffer,true);
            nifti_image *dofs = nifti_copy_nim_info(dof);
            dofs->nt = dofs->dim[4] = images->nt;
            dofs->nvox = dof->nvox*images->nt;
            dofs->data = (PrecisionTYPE *)calloc(dofs->nvox, dof->nbyper);
            PrecisionTYPE *intensityPtrD = static_cast<PrecisionTYPE *>(dofs->data);
            for(int t=0;t<images->nt;t++){
                char buffer[20];
                sprintf(buffer,"float%s%i.nii",style, t+1);
                nifti_image *dof = nifti_image_read(buffer,true);
                PrecisionTYPE *intensityPtrDD = static_cast<PrecisionTYPE *>(dof->data);
                int r=dof->nvox/3.0;
                for(int i=0;i<3;i++){
                    memcpy(&intensityPtrD[i*image->nt*r+t*r], &intensityPtrDD[i*r], dof->nbyper*r);	
                }		
                nifti_image_free(dof);
                remove(buffer); // delete spare floatcpp files
            }
            nifti_set_filenames(dofs,param->outputCPPName, 0, 0); // TODO NAME 	// write final dof data
            nifti_image_write(dofs);
            nifti_image_free(dofs);
        }
        else{
            std::string final_string = "";
            for(int t=0;t<images->nt;t++){
                char buffer[20];
                sprintf(buffer,"float%s%i.txt",style,t+1);
                std::ifstream ifs(buffer);  
                std::string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                final_string+=str; 
                remove(buffer);                
            }
            std::ofstream ofs(param->outputCPPName);
            ofs<<final_string.c_str();
        }
            
        // DELETE
        // delete: anchorx.nii floatx.nii outputResult.nii : I think this is all...
        remove("anchorx.nii");  // flakey...
        remove("floatx.nii");
        remove("outputResult.nii");
        remove("outputResult.nii.gz");
        
        // Write final image data
        nifti_set_filenames(image,param->outputResultName, 0, 0);
        nifti_image_write(image);
    }
	nifti_image_free(image);
    nifti_image_free(images);

	// CHECK CLEAN-UP    
	time_t end; time( &end );
	int minutes = (int)floorf(float(end-start)/60.0f);
	int seconds = (int)(end-start - 60*minutes);
	printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
	if(flag->localityFlag){printf("Registration to %i-local mean with %i iterations performed in %i min %i sec\n", param->locality, param->prinComp, minutes, seconds);}
    if(flag->singleFlag){printf("Single timepoint registration to image %i performed in %i min %i sec\n", param->singletimepoint, minutes, seconds);}
	if(flag->meanonly){printf("Registration to mean image with %i iterations performed in %i min %i sec\n", param->prinComp, minutes, seconds);}
    if(!flag->localityFlag & !flag->meanonly & !flag->singleFlag){printf("PPCNR registration with %i iterations performed in %i min %i sec\n", param->prinComp, minutes, seconds);}
	printf("Have a good day !\n");
    free( flag );
	free( param );
	return 0;
}
