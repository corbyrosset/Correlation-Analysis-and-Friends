function z = MyRbfpreimg_fpi(model)
% MYRBFPREIMG_FPI RBF pre-image problem by fixed-point iteration.

[dim,num_sv]=size(model.sv.X);
ker = 'rbf';
arg = model.options.arg;
%iXi = sum( model.sv.X.^2)';
%s2 = arg^2;

% Selection of the starting point out of the model.sv.X.
% The point in which is the objective function minimal is taken.
% Minimum over 50 randomly drawn points is used.
%--------------------------------------------------------------

rand_inx = randperm( num_sv );
rand_inx = rand_inx(1:min([num_sv,50]));
Z = model.sv.X(:,rand_inx);

%fval = kernel(Z,model.sv.X,ker,arg)*model.Alpha(:);
fval = myKernelMatrix(Z,model.sv.X,2,arg)*model.Alpha(:);
fval = -fval.^2;

[dummy, inx ] = min( fval );
z = Z(:,inx );

% Fixed point iteration.
%--------------------------------------
[z totIter finMove ecode] = myFixPtRbf(model.sv.X,model.Alpha,arg,z,500,1e-16);

%fprintf(1,'Total iterations = %d. Final movement = %f. Exit code = %d.\n',totIter,finMove,ecode);

return;