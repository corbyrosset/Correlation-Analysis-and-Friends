Kernel Canonical Correlation Analysis code for Matlab.

Author: Steven Van Vaerenbergh (http://gtas.unican.es/people/steven)


CONTENTS
========

- km_kcca_demo.m: demo file that performs KCCA on synthetically generated 
  data. Uses incomplete Cholesky decomposition.
- km_kcca_full_demo.m: demo file that performs KCCA on synthetically 
  generated data. Uses full kernel matrices.
- km_kernel.m: function that calculates kernel elements. Add your custom 
  kernels here.
- km_kernel_icd.m: function that calculates the incomplete Cholesky 
  decomposition of a kernel matrix. Does not require to calculate the 
  original kernel matrix.

If you use this code in your research please cite
@phdthesis {vanvaerenbergh2010kernel,
	author = {Van Vaerenbergh, Steven}
	title = {Kernel methods for nonlinear identification, equalization and separation of signals},
	year = {2010},
	school = {University of Cantabria},
	month = feb,
	note = {Software available at \url{http://sourceforge.net/projects/kmbox/}}
}

LICENSE
=======

This program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free 
Software Foundation, version 3 (http://www.gnu.org/licenses).


TOOLBOX
=======

This package is part of the Kernel Methods Toolbox. 
http://sourceforge.net/p/kmbox 

Check the toolbox' web site for newer versions of the included files.
