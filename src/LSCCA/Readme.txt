The LS_CCA package provides the Matlab implementations of Canonical Correlation Analysis, kernel CCA and their equivalent least squares formulations LS_CCA and KLS_CCA. By constructing a specific target matrix from the label information, it is proved that CCA is equivalent to the LS_CCA formulation. Several extensions of LS_CCA based on regularization are also implemented, such as the sparse LS_CCA formulation using 1-norm regularization. This package also provides the implementations of Orthonormalized Partial Least Squares (OPLS).

Specifically, this package implements the following techniques:
1. CCA.
2. Kernel CCA.
3. Least squares.
4. Kernel Least squares.
5. LS_CCA, the equivalent least squares formulation for CCA.
6. KLS_CCA, the equivalent kernel least squares formulation for kernel CCA.
7. Orthonormalized partial least squares.
For all techniques, the regularization is supported. For least squares formulations, LS_CCA, and KLS_CCA, 1-norm and 2-norm regularization are supported. 

In this package, there are two file folders, called "utilities" and "examples", respectively. All implementations of these techniques are in the folder "utilities". All scripts in the folder "examples" are designed to demonstrate the use of the functions provided in this package. For example, you can run CCA_example.m to run CCA function provided in this package. To run the example scripts, you can either put the example scripts and the functions in the folder "utilities" under the same directory, or you can add the directory of the "utilities" folder to your Matlab path.


The brief introduction to all files is listed below:
Package utilities:
CCA.m: 			it implements CCA with and without regularization.
colCenter.m: 		it centers the columns of a matrix, and it is used as subroutine.
KCCA.m: 		it implements kernel CCA with and without the regularization.
KLS.m: 			it implements kernel least squares.
KLS_CCA.m: 		it implements the equivalent kernel least squares formulation for kernel CCA.
l1_ls.m: 		it solves the lasso problem.
lasso_func.m: 		it is a general interface for lasso, thus the user can use other implementations of lasso by slightly revising this function.
LS.m: 			it implements the least squares formulation.
LS_CCA.m: 		it implements the equivalent least squares formulation for CCA.
OPLS.m: 		it implements orthonormalized partial least squares.
rowCenter.m: 		it centers the rows of a matrix, and it is used as subroutine.
solve_eigen.m: 		it solves a type of eigenvalue problem.

Package examples: 
CCA_example.m: 		it demonstrates how to use the CCA function.
KCCA_example.m: 	it demonstrates how to use KCCA function.
KLS_CCA_example.m: 	it demonstrates how to use KLS_CCA function.
LS_CCA_example.m: 	it demonstrates how to use the LS_CCA function.
OPLS_example.m		it demonstrates how to use the OPLS function.