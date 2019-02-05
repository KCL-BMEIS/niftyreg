// Author:  Lucas Fidon
// Class for F3D2 registration using the Ipopt optimisation solver
//

//#include "_reg_f3d2_ipopt.h"

//#include <cassert>
//#include <cstdio>

//using namespace Ipopt;

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//reg_f3d2_ipopt<T>::reg_f3d2_ipopt(int refTimePoint, int floTimePoint)
//  : reg_f3d2<T>::reg_f3d2(refTimePoint, floTimePoint){
  // make sure CumulativeExp is used and BCH is not used
//  this->useGradientCumulativeExp();
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//reg_f3d2_ipopt<T>::~reg_f3d2_ipopt(){
//#ifndef NDEBUG
//   reg_print_msg_debug("reg_f3d2_ipopt destructor called");
//#endif
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// returns the size of the problem.
//// IPOPT uses this information when allocating the arrays
//// that it will later ask you to fill with values.
//bool reg_f3d2_ipopt<T>::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
//				   Index& nnz_h_lag, IndexStyleEnum& index_style){
//  // number of spatial dimensions
//  int d = 0;
//  // number of variables =
//  // number of velocity vectors x number of dimension
//  n = 1.;
//  // number of constraints (null divergence of the velocity vector field)
//  m = 0.; // no constraint
//  //m = 1.;
//  if(this->inputReference->nx > 1){
//    ++d;
//    n *= this->inputReference->nx;
//    m *= this->inputReference->nx;
//  }
//  if(this->inputReference->ny > 1){
//    ++d;
//    n *= this->inputReference->ny;
//    m *= this->inputReference->ny;
//  }
//  if(this->inputReference->nz > 1){
//    ++d;
//    n *= this->inputReference->nz;
//    m *= this->inputReference->nz;
//  }
//  n *= d;

//  // number of non-zeros values in the jacobian of the constraint
//  nnz_jac_g = n * m; //full matrix

//  // number of non-zeros values in the hessian
//  nnz_h_lag = n * n; // full matrix

//  // use the C style indexing (0-based) for the matrices
//  index_style = TNLP::C_STYLE;

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// returns the variable bounds
//bool reg_f3d2_ipopt<T>::get_bounds_info(Index n, // number of variables (dim of x)
//                                        Number* x_l, // lower bounds for x
//                                        Number* x_u, // upperbound for x
//                                        Index m, // number of constraints (dim of g(x))
//                                        Number* g_l, // lower bounds for g(x)
//                                        Number* g_u){ // upper bounds for g(x)
//  // lower and upper bounds for the primal variables
//  for (Index i=0; i<n; i++){
//    x_l[i] = -2e19;  // -infty
//    x_u[i] = 2e19;  // +infty
//  }

//  // lower and upper bounds for the inequality constraints
//  for (Index i=0; i<m; i++){
//    g_l[i] = 0.;
//    g_u[i] = 0.;
//  }

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// returns the initial point for the problem
//bool reg_f3d2_ipopt<T>::get_starting_point(Index n, bool init_x, Number* x,
//					 bool init_z, Number* z_L, Number* z_U,
//					 Index m, bool init_lambda,
//					 Number* lambda)
//{
//  // Here, we assume we only have starting values for x, if you code
//  // your own NLP, you can provide starting values for the dual variables
//  // if you wish
//  assert(init_x == true);
//  assert(init_z == false);
//  assert(init_lambda == false);

//  // initialize to the given starting point
//  for (Index i=0; i<n; i++)
//  {
//    x[i] = 0.; // start with a null velocity vector field
//  }

//#ifdef skip_me
//  /* If checking derivatives, if is useful to choose different values */
//#endif

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// returns the value of the objective function
//bool reg_f3d2_ipopt<T>::eval_f(Index n, const Number* x,
//			     bool new_x, Number& obj_value)
//{
//  // HERE COMPUTE VALUE OF OBJECTIVE FUNCTION
//  obj_value = 0.;
//  // set the current velocity field to x
//  this->controlPointGrid->data = x;
//  // compute the objective function value
//  // take the opposite of objective value to maximise...
//  obj_value = -this->GetObjectiveFunctionValue();

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// return the gradient of the objective function grad_{x} f(x)
//// it sets all values of gradient in grad_f
//bool reg_f3d2_ipopt<T>::eval_grad_f(Index n, const Number* x,
//				  bool new_x, Number* grad_f){
//  // set the current velocity field to x
//  //this->setControlPointGridImage(vel);
//  this->controlPointGrid->data = x;
//  // compute the objective function gradient value
//  this->GetObjectiveFunctionGradient();
//  // combine forward and backward gradients
//  reg_tools_addImageToImage(this->transformaitionGradient, // in1
//                            this->backwardTransformationGradient, // in2
//                            this->transformationGradient); // out
//  // update the Ipopt gradient
//  *grad_f = *(this->transformationGradienti->data);

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// return the value of the constraints: g(x)
//// Divergence fo the velocity vector field
//bool reg_f3d2_ipopt<T>::eval_g(Index n, const Number* x,
//			     bool new_x, Index m, Number* g)
//{
//  // HERE COMPUTE VALUE OF ALL CONSTRAINTS IN g
//  for (Index j=0; j<m; j++){
//   g[j] = 0.;
//  }

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//// return the structure or values of the jacobian
//bool reg_f3d2_ipopt<T>::eval_jac_g(Index n, const Number* x, bool new_x,
//				 Index m, Index nele_jac, Index* iRow,
//				 Index *jCol, Number* values)
//{
//  if (values == NULL) {
//    // return the structure of the jacobian

//    // HERE FILL iRow and jCol
//    //for (Index j=0; j<m; j++){
//    //  iRow[3*j] = j;     jCol[3*j] = j;
//    //  iRow[3*j + 1] = j; jCol[3*j + 1] = j + 1;
//    //  iRow[3*j + 2] = j; jCol[3*j + 2] = j + 2;
//    //}
//  }
//  else {
//    // return the values of the jacobian of the constraints

//    // HERE FILL values
//    //for (Index j=0; j<m; j++){
//    //  values[3*j] = -1.;
//    //  values[3*j + 1] = (2.*x[j+1] + 1.5)*cos(x[j+2]);
//    //  values[3*j + 2] = -(x[j+1]*x[j+1] + 1.5*x[j+1] - a_[j])*sin(x[j+2]);
//    //}
//  }

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
////return the structure or values of the hessian
//bool reg_f3d2_ipopt<T>::eval_h(Index n, const Number* x, bool new_x,
//			     Number obj_factor, Index m, const Number* lambda,
//			     bool new_lambda, Index nele_hess, Index* iRow,
//			     Index* jCol, Number* values)
//{
//  if (values == NULL) {
//    // return the structure (lower or upper triangular part) of the
//    // Hessian of hte Lagrangian function

//    // HERE FILL iRow and jCol

//  }
//  else {
//    // return the values of the Hessian of hte Lagrangian function

//    // HERE FILL values
//  }

//  return true;
//}

/* *************************************************************** */
/* *************************************************************** */
//template <class T>
//void reg_f3d2_ipopt<T>::finalize_solution(SolverReturn status,
//					Index n, const Number* x,
//					const Number* z_L, const Number* z_U,
//					Index m, const Number* g,
//					const Number* lambda,
//					Number obj_value,
//					const IpoptData* ip_data,
//					IpoptCalculatedQuantities* ip_cq)
//{
//  // here is where we would store the solution to variables, or write
//  // to a file, etc so we could use the solution.

//  // update the current velocity vector field
//  // this->controlPointGrid->data = x;

//  // Save the control point image
//  //nifti_image *outputControlPointGridImage = REG->GetControlPointPositionImage();
//  //if(outputCPPImageName==NULL) outputCPPImageName=(char *)"outputCPP.nii";
//  //  memset(outputControlPointGridImage->descrip, 0, 80);
//  //  strcpy (outputControlPointGridImage->descrip,"Control point position from NiftyReg (reg_f3d)");
//  //if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
//  //  strcpy (outputControlPointGridImage->descrip,"Velocity field grid from NiftyReg (reg_f3d2)");
//  //reg_io_WriteImageFile(outputControlPointGridImage,outputCPPImageName);
//  //nifti_image_free(outputControlPointGridImage);
//  //outputControlPointGridImage=NULL;

//  printf("\nWriting solution file solution.txt\n");
//  FILE* fp = fopen("solution.txt", "w");

//  // For this example, we write the solution to the console
//  //fprintf(fp, "\n\nSolution of the primal variables, x\n");
//  //for (Index i=0; i<n; i++) {
//  //  fprintf(fp, "x[%d] = %e\n", i, x[i]);
//  //}

//  //fprintf(fp, "\n\nSolution of the bound multipliers, z_L and z_U\n");
//  //for (Index i=0; i<n; i++) {
//  //  fprintf(fp, "z_L[%d] = %e\n", i, z_L[i]);
//  //}
//  //for (Index i=0; i<n; i++) {
//  //  fprintf(fp, "z_U[%d] = %e\n", i, z_U[i]);
//  //}

//  // print final objective value
//  fprintf(fp, "\n\nObjective value\n");
//  fprintf(fp, "f(x*) = %e\n", obj_value);
//  fclose(fp);
//}
