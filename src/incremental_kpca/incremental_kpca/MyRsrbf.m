function red_model = MyRsrbf(model,options)
% MYRSRBF Reduced Set Method for RBF kernel expansion. Adapted from rsrbf.m
% of the STPR Toolbox (Franc and Hlavac).
%
% Synopsis:
%  red_model = MyRsrbf(model)
%  red_model = MyRsrbf(model,options)
%
% Description:
%  red_model = MyRsrbf(model) searchs for a kernel expansion
%    with nsv vectors which best approximates the input 
%    expansion given in model [Schol98a]. The Radial Basis 
%    kernel (RBF) is assumed (see 'help kernel').
%    
%  red_model = MyRsrbf(model,options) allows to specify the 
%    control paramaters.
%
% Input:
%  model [struct] Kernel expansion:
%   .Alpha [nsv x 1] Weight vector.
%   .sv.X [dim x nsv] Vectors defining the expansion.
%   .InitPreimg [dim x nip] Initial preimages that must appear in reduced 
%     set expansion.
%   .options.ker [string] Must be equal to 'rbf'.
%   .options.arg [1x1] Kernel argument (see 'help kernel').
% 
%  options [struct] Control parameters:
%   .nsv [1x1] Desired number of vectors in the reduced 
%     expansion (default round(length(model.Alpha)/2)).
%   .eps [1x1] Desired limit on the norm of difference between 
%     the original  normal vector and the reduced the normal 
%     vector in the  feature space. The algorithm is stopped 
%     when a lower  difference is achived (default 1e-6).
%   .preimage [string] Function called to solve the RBF pre-image 
%     problem (default 'rbfpreimg');
%   .verb [1x1] If 1 then progress info is display (default 0).
% 
% Output:
%  red_model [struct] Reduced kernel expansion.
%

% process inputs
%--------------------------------------------------
%if nargin < 2, options=[]; else options = c2s(options); end
if nargin < 2
    options=[];
end
if ~isfield(options,'nsv'), options.nsv = round(length(model.Alpha)/2); end
if ~isfield(options,'eps'), options.eps = 1e-6; end
if ~isfield(options,'preimage'), options.preimage = 'rbfpreimg2'; end
if ~isfield(options,'verb'), options.verb = 0; end

% init
%--------------------------------------------------
Z=[];
Beta = [];
Alpha = model.Alpha(:);
X = model.sv.X;
%Const = knorm(X,Alpha,model.options.ker,model.options.arg)^2;
Const = Alpha'*myGaussianKernelMatrix(X,X,model.options.arg)*Alpha;
error = inf;
iter = 0;

% check for and process initial preimages
%--------------------------------------------------
if (( isfield(model,'InitPreimg') )&&( ~isempty(model.InitPreimg) ))
    if options.verb, 
        fprintf(1,'Processing initial preimages...  ');
    end
    Z = model.InitPreimg;
    
    %---------------------
    % Using STPR toolbox.
    %Kz = kernel( Z, model.options.ker, model.options.arg );
    %Kzs = kernel( Z, model.sv.X, model.options.ker, model.options.arg );    
    %---------------------    

    Kz = myGaussianKernelMatrix(Z,Z,model.options.arg);
    Kzs = myGaussianKernelMatrix(Z,model.sv.X,model.options.arg);
  
    if (rcond(Kz) < 1e-6)
        Beta = pinv(Kz)*Kzs*model.Alpha(:);      
    elseif (isnan(rcond(Kz)) == 1)
        fprintf(1,'Condition number is NaN. ');
        red_model.Alpha = NaN;
        return;
    else      
        Beta = inv(Kz)*Kzs*model.Alpha(:);
    end
    error = GetAng(model.sv.X,model.Alpha,Z,Beta,model.options.ker,model.options.arg);
    if options.verb, 
        fprintf('ang(A,Z) = %f\n', error);
    end
    Alpha = [model.Alpha(:); -Beta(:)]';
    X = [model.sv.X, Z];
end

% main loop
%--------------------------------------------------
while error > options.eps & iter < options.nsv,
  
  iter = iter + 1;

  if options.verb, 
    fprintf('Iteration %d: ', iter);
  end
  
  tmp_model.Alpha = Alpha;
  tmp_model.sv.X = X;
  tmp_model.options = model.options;

  if options.verb, 
     fprintf('computing preimage, ');
  end
  Z = [Z, real(feval( options.preimage,tmp_model))];
  
  %---------------------
  % Using STPR toolbox.  
  %Kz = kernel( Z, model.options.ker, model.options.arg );
  %Kzs = kernel( Z, model.sv.X, model.options.ker, model.options.arg );
  %---------------------
  
  Kz = myGaussianKernelMatrix(Z,Z,model.options.arg);
  Kzs = myGaussianKernelMatrix(Z,model.sv.X,model.options.arg);
      
  if (rcond(Kz) < 1e-6)
      Beta = pinv(Kz)*Kzs*model.Alpha(:);      
  elseif (isnan(rcond(Kz)) == 1)
      fprintf(1,'Condition number is NaN. ');
      red_model.Alpha = NaN;
      return;
  else      
      Beta = inv(Kz)*Kzs*model.Alpha(:);
  end
 
  error = GetAng(model.sv.X,model.Alpha,Z,Beta,model.options.ker,model.options.arg);
 
  if options.verb, 
     fprintf('ang(A,Z) = %f\n', error);
  end

  Alpha = [model.Alpha(:); -Beta(:)]';
  X = [model.sv.X, Z];
  
end

red_model.options = model.options;
red_model.Alpha = Beta;
red_model.sv.X = Z;
red_model.nsv = size(Z,2);

return;
%EOF

%----------------------------------------------------------    
% Function to compute angle between 2 feature space vector.
%----------------------------------------------------------
function ANG = GetAng(A,ACOFF,B,BCOFF,KTYPE,KPARAM)

    norm_a = sqrt((ACOFF')*myGaussianKernelMatrix(A,A,KPARAM)*ACOFF);
    norm_b = sqrt((BCOFF')*myGaussianKernelMatrix(B,B,KPARAM)*BCOFF);
    ANG = acos(((ACOFF')*myGaussianKernelMatrix(A,B,KPARAM)*BCOFF)/(norm_a*norm_b));