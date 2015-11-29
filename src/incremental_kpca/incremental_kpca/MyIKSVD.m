function [ MODEL ] = MyIKSVD( A, OPTIONS )
%MYIKSVD Perform approximate incremental kernel SVD.
%----------------
% Parameter list:
%----------------
% A             = Data matrix.
% OPTIONS       = Training options structure.
%   .R          = Subspace dimension to keep.
%   .MAXLIB     = Maximum library size before compression.
%   .INC        = Number of new vectors per increment.
%   .KTYPE      = Kernel function type.
%   .KPARAM     = Parameter for KTYPE kernel function.
%   .DISP       = Display result in figure (for 2D data only).
%----------------------
% Output argument list:
%----------------------
% MODEL         = Output data structure.
%   .VECT       = Expansion vectors for kernel svd basis.
%   .ALPHA      = Expansion coefficients for kernel svd basis.
%   .SIGMA      = Singular values.
%   .ABORTION   = Update abortion counter.
%   .OPTIONS    = Original training options.

SECPASS = 1; % Second pass of coefficient estimation.
ORTHO = 1; % Orthogonalize.

%--------------------------
% Process input parameters.
%--------------------------
if (nargin == 1)||...
   (~isfield(OPTIONS,'R'))||...
   (~isfield(OPTIONS,'MAXLIB'))||...
   (~isfield(OPTIONS,'INC'))||...
   (~isfield(OPTIONS,'KTYPE'))||...
   (~isfield(OPTIONS,'KPARAM'))||...
   (~isfield(OPTIONS,'DISP'))
        fprintf(1,'Required parameters missing.\n');
        OPTIONS = CreateOpt(2);
end

%----------------------
% Get training options.
%----------------------
R = OPTIONS.R;
MAXLIB = OPTIONS.MAXLIB;
INC = OPTIONS.INC;
KTYPE = OPTIONS.KTYPE;
KPARAM = OPTIONS.KPARAM;
DISP = OPTIONS.DISP;

%----------------------
% Get number of points.
%----------------------
[ d N ] = size(A);

%----------------------
% Perform initial KSVD.
%----------------------
[ model ] = MyKSVD( A(:,1:MAXLIB), struct('R',MAXLIB,'KTYPE',KTYPE,'KPARAM',KPARAM,'DISP',0) );
cR = min(R,size(model.ALPHA,2));
dataset = repmat( model.VECT, 1, cR );
alpha = zeros( MAXLIB*cR, cR );
for r=1:cR
    alpha( ((r-1)*MAXLIB+1):(r*MAXLIB), r ) = model.ALPHA(:,r);
end
sigma = model.SIGMA(1:cR,1:cR);

%-------------
% Start timer.
%-------------
tic;

%--------------------------
% Perform incremental KPCA.
%--------------------------
abortion = 0;

for n = (MAXLIB+1):INC:( N - mod(N-MAXLIB,INC) )
    
    % Current database size.
    dsize = size(dataset,2);
    
    % Current number of basis vectors.
    cR = size(alpha,2);
    
    % New vectors.
    B = A(:,n:(n+INC-1));
    
    %-------------
    % Start IKSVD.
    %-------------
    L = alpha'*myKernelMatrix( dataset, B, KTYPE, KPARAM );
    betaprime = [ -alpha*L ; eye(INC) ];
            
    %--------------------------
    % Safety feature- abortion.
    %--------------------------    
    Mh = betaprime'*myKernelMatrix( [ dataset B ], [ dataset B ], KTYPE, KPARAM )*betaprime;
    rankM = rank(Mh);
    if (rankM==0)
       warning('Mh very rank deficient. Update aborted.');
       abortion = abortion + 1;
       continue;
    end

    %---------------------------
    % KSVD to replace kernel QR.
    %---------------------------
    [Qh Dh] = eig(Mh);
    Dh = abs(Dh);
    [sorted sortind] = sort(diag(Dh),1,'descend');
    Dh = diag(sorted);         
    Qh = real(Qh(:,sortind));
    
    Ohm = betaprime*Qh(:,1:rankM)*diag(1./sqrt(diag(Dh(1:rankM,1:rankM)))); % Take first-rankM orthogonal vectors only.
    K = diag(sqrt(diag(Dh)))*(Qh');

    %--------------------
    % Construct matrix F.
    %--------------------
    F = [ sigma L ; zeros(rankM,cR) K(1:rankM,:) ];
    [Uf Sf Vf] = svd(F);

    %-----------------------------
    % Update left singular values.
    %-----------------------------
    Psi = [ alpha ; zeros(INC,cR) ]*Uf(1:cR,:) + Ohm*Uf((cR+1):(cR+rankM),:);
    
    %%%%% Check validity of all matrices %%%%%
    if ((myIsvalid(L         ) ~= 0)||...
        (myIsvalid(betaprime ) ~= 0)||...
        (myIsvalid(Mh        ) ~= 0)||...
        (myIsvalid(Ohm       ) ~= 0)||...
        (myIsvalid(K         ) ~= 0)||...
        (myIsvalid(Psi       ) ~= 0))
        warning('One or more matrices are invalid.');
    end
    
    %--------------
    % Updated KSVD.
    %--------------
    dataset = [ dataset B ];
    if ( (cR+rankM) < R)
        alpha = Psi;
        sigma = Sf(1:(cR+rankM),1:(cR+rankM));
    else
        alpha = Psi(:,1:R);
        sigma = Sf(1:R,1:R);
    end
    
    %---------------------------
    % Size of current expansion.
    %---------------------------
    [ r_alp c_alp ] = size(alpha);    
    fprintf(1,'Number of vectors processed = %d.\n',n+INC-1-abortion*INC);
    fprintf(1,'    Current library size = %d.\n',r_alp);
    
    %---------------------------------
    % Construct reduced set expansion.
    %---------------------------------
    if ( r_alp > (MAXLIB*R))
        fprintf(1,'    Compressing...\n');
        %[ Z, BETA ] = SimCompress( dataset, alpha, MAXLIB, KTYPE, KPARAM );
        [ Z, BETA ] = IterCompress( dataset, alpha, MAXLIB, KTYPE, KPARAM );
        [rZ cZ] = size(Z);
        if ((rZ==1)&&(cZ==1)&&(isnan(Z)==1))
            fprintf(1,'    compression aborted.\n');
        else
            if (SECPASS == 1)
                fprintf(1,'    Second coefficient estimation...\n');
                [ Z, BETA ] = FillRs( dataset,alpha,Z,BETA,KTYPE,KPARAM );
                fprintf(1,'    Second coefficient estimation done.\n');
            end            
            if (ORTHO == 1)
                %-------------------------------
                % Re-orthogonalize RS expansion.
                %-------------------------------
                fprintf(1,'    Reorthogonalizing...\n');
                % Perform svd on approximated basis.
                Mo = (BETA(:,1:size(alpha,2))')*myKernelMatrix(Z,Z,KTYPE,KPARAM)*BETA(:,1:size(alpha,2));
                [Qo Do] = eig(Mo);
                if sum(sum(Do<0)) > 0        
                    Do = abs(Do);
                end
                newBeta = BETA(:,1:size(alpha,2))*Qo*diag(1./sqrt(diag(Do)));            
                % Project approximate basis onto its orthogonalized version.
                proj = (newBeta')*myKernelMatrix(Z,Z,KTYPE,KPARAM)*BETA(:,1:size(alpha,2));
                % Normalize.
                proj = proj*diag(1./sqrt(sum(proj.^2)));
                % Rotate svd basis to coincide with the approximated basis.
                newBeta = newBeta*proj;
                % Project original basis onto orthogonalized approximate basis.
                sigproj = (newBeta')*myKernelMatrix(Z,dataset,KTYPE,KPARAM)*alpha*sigma;
                % Find equivalent sigma.
                newSigma = diag(diag(sigproj));
                % Store new data and coefficients.
                BETA = newBeta;
                sigma = newSigma;
            end
            % Get approximation error here.
            norm_a = sqrt((alpha')*myKernelMatrix(dataset,dataset,KTYPE,KPARAM)*alpha);    
            norm_b = sqrt((BETA')*myKernelMatrix(Z,Z,KTYPE,KPARAM)*BETA);    
            dotprod = (alpha')*myKernelMatrix(dataset,Z,KTYPE,KPARAM)*BETA;            
            angdiff = acos(diag(dotprod)./(diag(norm_a).*diag(norm_b)));
            fprintf(1,'    Final approximation errors:\n');
            for diff = 1:size(BETA,2)
                fprintf(1,'        vector %d, error = %f radians.\n',diff,angdiff(diff));
            end
            % Update storage here.            
            dataset = Z;
            alpha = BETA;
        end
    else
        fprintf(1,'    No compression.\n');
    end
    
end

%------------------
% Get elapsed time.
%------------------
fprintf(1,'Elapsed time is %f.\n',toc);

%---------------------
% Project and display.
%---------------------
if ((DISP == 1)&&(d == 2))

    fprintf(1,'Plotting results...');
    
    % Get test points.
    [X Y] = meshgrid(-1:0.05:1, -0.5:0.05:1.5);    
    [rX cX] = size(X);    
    Pts_x = reshape(X,1,rX*cX);
    Pts_y = reshape(Y,1,rX*cX);
    Pts = [ Pts_x ; Pts_y ];  
    
    % Number of data points.
    [ dPts nPts ] = size(Pts);

    % Project onto subspaces.
    proj = (alpha(:,1:R)')*myKernelMatrix(dataset,Pts,KTYPE,KPARAM);       

    for pc = 1:R
           
        % Display input points.
        figure;
        set(gcf,'Renderer','zbuffer');
        axis([-1 1 -0.5 1.5]);

        % Contour values.
        Z = reshape(proj(pc,:),rX,cX); 
        projcon(pc).Z = Z;        

        % Plot.
        colormap(gray);
        hold on;
        pcolor(X, Y, Z);
        shading interp;
        contour(X, Y, Z, 9, 'b');    
        plot(A(1,:), A(2,:), 'r.');
        box on;
        
        drawnow;
        
    end
    
    fprintf(1,'Done.\n');
 
else
    projcon = 0;
end


%----------------------------
% Construct return structure.
%----------------------------
MODEL = struct(...
'VECT',dataset,...
'ALPHA',alpha,...
'SIGMA',sigma,...
'ABORTION',abortion,...
'OPTIONS',OPTIONS,...
'PROJ',projcon...
);

%-----------------------------------------
% Funnction to check validity of matrices.
%-----------------------------------------
function V = myIsvalid(M)
    V = 0;
    if (sum(sum(isnan(M)))>0)
        V = V + 1;
    end
    if (sum(sum(isinf(M)))>0)
        V = V + 1;
    end
    if (isreal(M)~=1)
        V = V + 1;
    end