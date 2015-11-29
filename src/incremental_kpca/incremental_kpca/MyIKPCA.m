function [ MODEL ] = MyIKPCA( A, OPTIONS )
%MYIKPCA Perform approximate incremental kernel PCA via incremental kernel SVD.
%----------------
% Parameter list:
%----------------
% A             = Data matrix.
% OPTIONS       = Training options.
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
%   .VECT       = Expansion vectors for kernel principal components.
%   .ALPHA      = Expansion coefficients for kernel principal components.
%   .NU         = Expansion coefficients for data mean.
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
% Perform initial KPCA.
%----------------------
[ model ] = MyKPCA( A(:,1:MAXLIB), struct('R',MAXLIB,'KTYPE',KTYPE,'KPARAM',KPARAM,'DISP',0) );
cR = min(R,size(model.ALPHA,2));
dataset = repmat( model.VECT, 1, cR+1 );
alpha = zeros( MAXLIB*(cR+1), cR );
for r=1:cR
    alpha( ((r-1)*MAXLIB+1):(r*MAXLIB), r ) = model.ALPHA(:,r);
end
nu = zeros( MAXLIB*(cR+1), 1 );
nu( (cR*MAXLIB+1):((cR+1)*MAXLIB), 1 ) = model.NU;
sigma = model.SIGMA(1:cR,1:cR);

%-------------
% Start timer.
%-------------
tic;

%--------------------------
% Perform incremental KPCA.
%--------------------------
abortion = 0;
h = waitbar(0,'IKPCA...');

for n = (MAXLIB+1):INC:( N - mod(N-MAXLIB,INC) )
    
    % tic;
    
    % Current database size.
    dsize = size(dataset,2);
    
    % Current number of principal components.
    cR = size(alpha,2);
    
    % New vectors.
    B = A(:,n:(n+INC-1));
    
    % Compute omega.
    %omega = 1;
    omega = (1/INC)*ones(INC,1);
    
    % Compute nubar.
    %nubar = (1/(n-1-abortion + 1))*[ (n-1-abortion)*nu ; omega ];
    nubar = (1/(n-1-abortion*INC + INC))*[ (n-1-abortion*INC)*nu ; INC*omega ];
    
    % Compute omega prime.
    %omegaprime = 0;
    omegaprime = eye(INC) - omega*ones(1,INC);
    
    % Compute gamma.
    %gamma = [ [ zeros(dsize,1) ; omegaprime ]  sqrt(((n-1-abortion)*1)/(n-1-abortion + 1))*[ nu ; -omega ] ];
    gamma = [ [ zeros(dsize,INC) ; omegaprime ]  sqrt(((n-1-abortion*INC)*INC)/(n-1-abortion*INC + INC))*[ nu ; -omega ] ];
    
    %-------
    % Get L.
    %-------
    L = (alpha')*myKernelMatrix( dataset, [ dataset B ], KTYPE, KPARAM )*gamma;
        
    %-------------
    % Get J and K.
    %-------------

    % Compute beta.
    %beta = [ gamma(1:dsize,:) - alpha*L ; gamma(dsize+1,:) ];
    beta = [ gamma(1:dsize,:) - alpha*L ; gamma((dsize+1):(dsize+INC),:) ];

    % Compute Mh.
    Mh = (beta')*myKernelMatrix( [ dataset B ], [ dataset B ], KTYPE, KPARAM )*beta;
    
    %--------------------------
    % Safety feature- abortion.
    %--------------------------    
    rankM = rank(Mh);
    if (rankM==0)
       warning('Mh very rank deficient. Update aborted.');
       abortion = abortion + 1;
       continue;
    end
    
    [Qh Dh] = eig(Mh);
    if sum(sum(Dh<0)) > 0
        %warning('Some eigenvalues are negative! Taking absolute values.');
        Dh = abs(Dh);
    end
    [sorted sortind] = sort(diag(Dh),1,'descend');
    Dh = diag(sorted);         
    Qh = real(Qh(:,sortind));   

    Ohm = beta*Qh(:,1:rankM)*diag(1./sqrt(diag(Dh(1:rankM,1:rankM)))); % Take first-rankM orthogonal vectors only.
    K = diag(sqrt(diag(Dh)))*(Qh');

    %--------------------
    % Construct matrix F.
    %--------------------
    F = [ sigma L ; zeros(rankM,cR) K(1:rankM,:) ];
    [Uf Sf Vf] = svd(F);

    %-----------------------------
    % Update left singular values.
    %-----------------------------
    %Psi = [ alpha ; zeros(1,cR) ]*Uf(1:cR,:) + Ohm*Uf((cR+1):(cR+rankM),:);
    Psi = [ alpha ; zeros(INC,cR) ]*Uf(1:cR,:) + Ohm*Uf((cR+1):(cR+rankM),:);
    
    %%%%% Check validity of all matrices %%%%%
    if ((myIsvalid(omega     ) ~= 0)||...
        (myIsvalid(nubar     ) ~= 0)||...
        (myIsvalid(omegaprime) ~= 0)||...
        (myIsvalid(gamma     ) ~= 0)||...
        (myIsvalid(L         ) ~= 0)||...
        (myIsvalid(beta      ) ~= 0)||...
        (myIsvalid(Mh        ) ~= 0)||...
        (myIsvalid(Ohm       ) ~= 0)||...
        (myIsvalid(K         ) ~= 0)||...
        (myIsvalid(Psi       ) ~= 0))
        warning('One or more matrices are invalid.');
    end
    
    %--------------
    % Updated KPCA.
    %--------------
    dataset = [ dataset B ];
    if ( (cR+rankM) < R)
        alpha = Psi;
        sigma = Sf(1:(cR+rankM),1:(cR+rankM));
    else
        alpha = Psi(:,1:R);
        sigma = Sf(1:R,1:R);
    end
    nu = nubar;
    
    %---------------------------
    % Size of current expansion.
    %---------------------------
    [ r_alp c_alp ] = size(alpha);    
    %fprintf(1,'Number of vectors processed = %d.\n',n-abortion);
    fprintf(1,'Number of vectors processed = %d.\n',n+INC-1-abortion*INC);
    fprintf(1,'    Current library size = %d.\n',r_alp);
    
    %---------------------------------
    % Construct reduced set expansion.
    %---------------------------------
    if ( r_alp > (MAXLIB*(R+1)) )
        fprintf(1,'    Compressing...\n');
        %[ Z, BETA ] = SimCompress( dataset, [alpha nu], MAXLIB, KTYPE, KPARAM );
        [ Z, BETA ] = IterCompress( dataset, [alpha nu], MAXLIB, KTYPE, KPARAM );
        [rZ cZ] = size(Z);
        if ((rZ==1)&&(cZ==1)&&(isnan(Z)==1))
            fprintf(1,'    Compression aborted.\n');
        else
            fprintf(1,'    Compression done.\n');                        
            if (SECPASS == 1)
                fprintf(1,'    Second coefficient estimation...\n');
                [ Z, BETA ] = FillRs( dataset,[alpha nu],Z,BETA,KTYPE,KPARAM );            
                fprintf(1,'    Second coefficient estimation done.\n');
            end
            if (ORTHO == 1)
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
                fprintf(1,'    Reorthogonalization done.\n');

                % dataset = Z;
                % alpha = newBeta;
                % nu = BETA(:,size(alpha,2)+1);
                % sigma = newSigma;
                BETA = [ newBeta BETA(:,size(alpha,2)+1) ];
                sigma = newSigma;
            end
            % Get approximation error here.
            norm_a = sqrt(([alpha nu]')*myKernelMatrix(dataset,dataset,KTYPE,KPARAM)*[alpha nu]);    
            norm_b = sqrt((BETA')*myKernelMatrix(Z,Z,KTYPE,KPARAM)*BETA);    
            dotprod = ([alpha nu]')*myKernelMatrix(dataset,Z,KTYPE,KPARAM)*BETA;            
            angdiff = acos(diag(dotprod)./(diag(norm_a).*diag(norm_b)));
            fprintf(1,'    Final approximation errors:\n');
            for diff = 1:size(BETA,2)
                fprintf(1,'        vector %d, error = %f radians.\n',diff,angdiff(diff));
            end
            % Update storage here.            
            dataset = Z;
            alpha = BETA(:,1:size(alpha,2));
            nu = BETA(:,size(alpha,2)+1);
        end
    else
        fprintf(1,'    No compression.\n');
    end
    waitbar(n/(N - mod(N-MAXLIB,INC)));

    
end
close(h);

%------------------
% Get elapsed time.
%------------------
toc;

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

    % Mean adjust data.
    muadj = [ (-nu)*ones(1,nPts) ; eye(nPts) ];

    % Project onto subspaces.
    proj = (alpha(:,1:R)')*myKernelMatrix(dataset,[dataset Pts],KTYPE,KPARAM)*muadj;            

    for pc = 1:R
        
        % Display input points.        
        figure;        
        set(gcf, 'Renderer', 'zbuffer');        
        axis([-1 1 -0.5 1.5]);

        % Contour values.
        if pc<3
            Z = reshape(-proj(pc,:),rX,cX);
        else
            Z = reshape(proj(pc,:),rX,cX);
        end
       
        % Plot
        colormap(gray);
        hold on;
        box on;
        axis off;
        pcolor(X, Y, Z);
        shading interp;
        contour(X, Y, Z, 9, 'b');    
        plot(A(1,:), A(2,:), 'r.')
        
        drawnow;
        
    end
    fprintf(1,'Done.\n');
 
end

%----------------------------
% Construct return structure.
%----------------------------
MODEL = struct(...
'VECT',dataset,...
'ALPHA',alpha,...
'NU',nu,...
'SIGMA',sigma,...
'ABORTION',abortion,...
'OPTIONS',OPTIONS...
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