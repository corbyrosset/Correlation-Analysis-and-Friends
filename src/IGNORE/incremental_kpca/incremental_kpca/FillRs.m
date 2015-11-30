function [ Zf, BETAf ] = FillRs( A,ALPHA,Z,BETA,KTYPE,KPARAM )
%FILLRS Fill empty coefficients for RS expansions.

% Check matrix sizes.
[mBETA nBETA] = size(BETA);
BETAf = zeros(mBETA,nBETA);

% Second pass.
% fprintf(1,'    Final approximation errors:\n');
for r = 1:nBETA
    
    diff1 = GetAng(A,ALPHA(:,r),Z,BETA(:,r),KTYPE,KPARAM);    
    Kz = myKernelMatrix( Z, Z, KTYPE, KPARAM );
    Kzs = myKernelMatrix( Z, A, KTYPE, KPARAM );    
    if (rcond(Kz) < 1e-6)
        newBETA = pinv(Kz)*Kzs*ALPHA(:,r);      
    elseif (isnan(rcond(Kz)) == 1)
        newBETA = BETA(:,r);
    else      
        newBETA = inv(Kz)*Kzs*ALPHA(:,r);
    end    
    diff2 = GetAng(A,ALPHA(:,r),Z,newBETA,KTYPE,KPARAM);    
    
    if diff2 < diff1        
        BETAf(:,r) = newBETA;        
    else
        BETAf(:,r) = BETA(:,r);        
    end
    %fprintf(1,'        vector %d, error = %f radians.\n',r,min(diff1,diff2));

end

Zf = Z;

return;

%----------------------------------------------------------    
% Function to compute angle between 2 feature space vector.
%----------------------------------------------------------
function ANG = GetAng(A,ACOFF,B,BCOFF,KTYPE,KPARAM)
    % norm_a = GetNorm(A,ACOFF,KTYPE,KPARAM);
    % norm_b = GetNorm(B,BCOFF,KTYPE,KPARAM);
    norm_a = sqrt((ACOFF')*myKernelMatrix(A,A,KTYPE,KPARAM)*ACOFF);    
    norm_b = sqrt((BCOFF')*myKernelMatrix(B,B,KTYPE,KPARAM)*BCOFF);    
    ANG = acos(((ACOFF')*myKernelMatrix(A,B,KTYPE,KPARAM)*BCOFF)/(norm_a*norm_b));        

% %-----------------------------------------------    
% % Function to compute feature space vector norm.
% %-----------------------------------------------
% function N = GetNorm(V,ALPHA,KTYPE,KPARAM)
%     N = sqrt((ALPHA')*myKernelMatrix(V,V,KTYPE,KPARAM)*ALPHA);