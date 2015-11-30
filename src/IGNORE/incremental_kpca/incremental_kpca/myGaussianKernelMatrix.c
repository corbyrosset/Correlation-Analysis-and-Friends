#include "mex.h"
#include "math.h"

//double myKernel(double *x, double *z, int pix, int type, double *param);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // Input/output variables.
  double *K;
  double *A1;
  double *A2;
  double *KPARAM;
  
  // Intermediate variables.
  int m1,n1,m2,n2;
  int i,j;
  
  // For Gaussian kernel evaluations.
  double accum;  
  int n;
  
  double sigsum;
  
  /*---------------------------------------*/
  /* Check for proper number of arguments. */
  /*---------------------------------------*/
  /*
  if (nrhs != 3) {
    mexErrMsgTxt("Three inputs required.");
  } else if (nlhs > 2) {
    mexErrMsgTxt("Too many output arguments.");
  }
  */
  
  /*------------------------*/
  /* Get input matrix sizes.*/
  /*------------------------*/
  // Assign pointers to inputs.
  A1 = mxGetPr(prhs[0]);
  A2 = mxGetPr(prhs[1]);
  KPARAM = mxGetPr(prhs[2]);
  // Get sizes of A matrices.
  m1 = mxGetM(prhs[0]);
  n1 = mxGetN(prhs[0]);
  m2 = mxGetM(prhs[1]);
  n2 = mxGetN(prhs[1]);
  
  //mexPrintf("Kernel param is %f.\n",KPARAM[0]);
  
  /*----------------------------------------*/
  /* Create matrix for the return argument. */
  /*----------------------------------------*/
  plhs[0] = mxCreateDoubleMatrix(n1,n2,mxREAL);   
  // Assign pointers to output.
  K = mxGetPr(plhs[0]);
  
  sigsum = (-1)/( 2*KPARAM[0]*KPARAM[0] );
  
  for(j=0;j<n1;j++)
  {
    for(i=0;i<n2;i++)
    {
        accum = 0;
        for(n=0;n<m1;n++)
        {
            //if ((i==0)&&(j==0)&&(n<10))
            //    mexPrintf("val = %f.\n",*(A2+i*m2+n));
            
            accum = accum + ( *(A1+j*m1+n) - *(A2+i*m2+n) )*( *(A1+j*m1+n) - *(A2+i*m2+n) );            
        }
        K[ j + i*n1 ] = exp( sigsum*accum );
    }     
  }
}

/*
double myKernel(double *x, double *z, int pix, int type, double *param)
{   
    int n;
    double accum, keval;
    
    switch(type)
    {
        case 1:
            // Polynomial kernel.
            accum = 0;
            for(n=0;n<pix;n++)
            {
                accum = accum + x[n]*z[n];
            }
            keval = pow(accum,param[0]);
            //mexPrintf("accum = %f.\n",accum);
            //mexPrintf("param[0] = %f.\n",param[0]);
            //mexPrintf("param[1] = %f.\n",param[1]);            
            //mexPrintf("keval = %f.\n",keval);
            break;
        case 2:
            // Gaussian kernel.
            accum = 0;
            for(n=0;n<pix;n++)
            {
                accum = accum + (x[n] - z[n])*(x[n] - z[n]);
            }
            keval = exp( (-1)*accum/(2*param[0]*param[0]) );
            break;
        case 3:
            // Epanechnikov kernel.
            accum = 0;
            for(n=0;n<pix;n++)
            {
                accum = accum + (x[n] - z[n])*(x[n] - z[n]);
            }
            if ((sqrt(accum)/param[0]) > 1)
            {
                keval = 0;
            }
            else
            {
                keval = 0.75*(1-accum/(param[0]*param[0]));
            }
            break;
        case 4:
            // Uniform kernel.
            accum = 0;
            for(n=0;n<pix;n++)
            {
                accum = accum + (x[n] - z[n])*(x[n] - z[n]);
            }
            if (sqrt(accum) > param[0])
            {
                keval = 0;
            }
            else
            {
                keval = 1;
            }
            break;
        default:
            // Linear kernel.
            accum = 0;
            for(n=0;n<pix;n++)
            {
                accum = accum + x[n]*z[n];
            }
            keval = pow(accum,1.0);
    }
    
    return(keval); 
}
 */