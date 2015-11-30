Implementation for Incremental Kernel Principal Component Analysis
-----------------------------------------------------------------
Copyright Tat-Jun Chin, May 2006.
tatjun@gmail.com

This code package is provided for non-commercial academic research use.
Commercial use is strictly prohibited without the author's written
consent.

Please cite the following paper if you use this code package or part of it
in your publication:
Tat-Jun Chin, D. Suter
Incremental Kernel Principal Component Analysis
IEEE Trans. on In Image Processing, Vol. 16, No. 6. (2007), pp. 1662-1674.

-------------
Preliminaries
-------------

This distribution of codes implement the following:
1. Kernel SVD (KSVD) in MyKSVD.m.
2. Kernel PCA (KPCA) in MyKPCA.m.
3. Incremental KSVD (IKSVD) in MyIKSVD.m.
4. Incremental KPCA (IKPCA) in MyIKPCA.m.

The other codes implement required auxiliary functions. In addition, in
some instances the 4 functions above require a few supporting functions
from the Statistical Pattern Recognition (STPR) Toolbox, which is available
from http://cmp.felk.cvut.cz/cmp/software/stprtool/index.html.

To ensure smooth running of the 4 functions, please download and install
the STPR Toolbox (and remember to add it to the Matlab search path).

The 4 functions also call the following C codes from WITHIN MATLAB:
1. myKernelMatrix.c
2. myGaussianKernelMatrix.c
This can be achieved easily by compiling them into Matlab dll's, which
then allows them to be called as though they are Matlab functions from
within Matlab. The corresponding dll's are:
1. myKernelMatrix.mexw32
2. myGaussianKernelMatrix.mexw32
Matlab dll's are version and platform dependent. It would be best to
recompile the C codes on your machine, as such:
>> mex myKernelMatrix.c
>> mex myGaussianKernelMatrix.c

----------
Quickstart
----------

Generate a synthetic 2D dataset with 1000 points:
>> x = GenStanToy(1000);

Process with KSVD:
>> model = MyKSVD(x,struct('R',3,'KTYPE',2,'KPARAM',1,'DISP',1));
You should see 3 figures which show the projection contours of the first-3
kernel subspace bases of x.

Process with incremental KSVD:
>> model = MyIKSVD(x,struct('R',3,'MAXLIB',10,'INC',30,'KTYPE',2,'KPARAM',1,'DISP',1));
You should see 3 figures which show the projection contours of the first-3
kernel subspace bases of x which are estimated incrementally.

Process with KPCA:
>> model = MyKPCA(x,struct('R',3,'KTYPE',2,'KPARAM',1,'DISP',1));
You should see 3 figures which show the projection contours of the first-3
kernel principal components of x.

Process with incremental KPCA:
>> model = MyIKPCA(x,struct('R',3,'MAXLIB',10,'INC',30,'KTYPE',2,'KPARAM',1,'DISP',1));
You should see 3 figures which show the projection contours of the first-3
kernel principal components of x which are estimated incrementally.
