function [ Z, BETA ] = IterCompress( A, ALPHA, ZNUM, KTYPE, KPARAM )
%SIMCOMPRESS Compress linear expansions of multiple feature space vectors 
%simultaneously via constructing RS expansions.
%----------------
% Parameter list:
%----------------
% A         = Data matrix.
% ALPHA     = Expansion coefficients.
% ZNUM      = Number of pre-images per feature vector.
% KTYPE     = Kernel function type.
% KPARAM    = Parameter for KTYPE kernel function.
%----------------------
% Output argument list:
%----------------------
% Z         = Pre-images set.
% BETA      = Expansion coefficients for pre-images.

% Check matrix sizes.
[mA nA] = size(A);
[mALPHA nALPHA] = size(ALPHA);
if (nA~=mALPHA)
    error('Incorrect matrix sizes!');
    Z = NaN;
    BETA = NaN;
    return;
end

% Get all RS expansions.
for r = 1:nALPHA    
    %-------------------------------------
    % The following uses the STPR toolbox.
    %-------------------------------------
    if (KTYPE == 2)
        
        model.Alpha = ALPHA(:,r);
        model.sv.X = A;
        if r>1
            model.InitPreimg = Z;
        end
        model.options.ker = 'rbf';
        model.options.arg = KPARAM;
        options.nsv = ZNUM;
        %options.preimage = 'rbfpreimg2';
        options.preimage = 'MyRbfpreimg_opt';
        %options.preimage = 'MyRbfpreimg_fpi';
        options.verb = 0;
        red_model = MyRsrbf(model,options);
        if (isnan(red_model.Alpha) == 1)            
            Z = NaN;
            BETA = NaN;
            return;
        end
        if r==1
            Z = red_model.sv.X;
            BETA = red_model.Alpha;
        else
            Z = red_model.sv.X;
            BETA = [ [ BETA ; zeros( size(red_model.Alpha,1)-size(BETA,1),r-1) ] red_model.Alpha ];
        end
        
    elseif ((KTYPE == 1)&&(KPARAM == 2))        
        
        model.Alpha = ALPHA(:,r);
        model.b = 0;
        model.sv.X = A;
        model.nsv = nA;
        model.options.ker = 'poly';
        model.options.arg = [2 0];
        red_model_1 = rspoly2(model,ZNUM);
        
        if r==1            
            Z = red_model_1.sv.X;
            BETA = red_model_1.Alpha;            
        else
            Z = [ Z red_model_1.sv.X ];
            BETA = [ [ BETA ; zeros( size(red_model_1.Alpha,1),r-1) ] [ zeros(size(BETA,1),1) ; red_model_1.Alpha ] ];
        end
    else
        error('Only RBF kernels and 2nd-degree polynomial kernal are supported!');
        return;        
    end
end