function [ MODEL ] = MyKSVD( A, OPTIONS )
%MYKSVD Kernel SVD codes.
%----------------
% Parameter list:
%----------------
% A             = Data matrix.
% OPTIONS       = Training options.
%   .R          = Subspace dimension to keep.
%   .KTYPE      = Kernel function type.
%   .KPARAM     = Parameter for KTYPE kernel function.
%   .DISP       = Display result in figure (for 2D data only).
%----------------------
% Output argument list:
%----------------------
% MODEL         = Output data structure.
%   .VECT       = Expansion vectors for kernel principal components.
%   .ALPHA      = Expansion coefficients for kernel principal components.
%   .SIGMA      = Singular values.
%   .OPTIONS    = Original training options.

%--------------------------
% Process input parameters.
%--------------------------
if (nargin < 2)||...
   (~isfield(OPTIONS,'R'))||...
   (~isfield(OPTIONS,'KTYPE'))||...
   (~isfield(OPTIONS,'KPARAM'))||...
   (~isfield(OPTIONS,'DISP'))
        fprintf(1,'Required parameters missing.\n');
        OPTIONS = CreateOpt(1);
end

%----------------------
% Get training options.
%----------------------
R = OPTIONS.R;
KTYPE = OPTIONS.KTYPE;
KPARAM = OPTIONS.KPARAM;
DISP = OPTIONS.DISP;

%----------------------
% Get number of points.
%----------------------
[ d n ] = size(A);

%-------------
% Start timer.
%-------------
% tic;

%----------------------
% Create kernel matrix.
%----------------------
M = myKernelMatrix( A, A, KTYPE, KPARAM );

%----------
% EVD of M.
%----------
[ Q D ] = eig(M);
D = real(D);
[sorted sortind] = sort(diag(D),1,'descend');
D = diag(sorted);
Q = Q(:,sortind);

%--------------------------
% Check rank of input data.
%--------------------------
if R > rank(D)
    R = rank(D);
    OPTIONS.R = rank(D);
    warning('Rank of data is less than R.');
end

%-------------------------------
% Retain R principal components.
%-------------------------------
ALPHA = Q(:,1:R)*diag(1./sqrt(diag(D(1:R,1:R))));

%-------------------------
% Singular values of ksvd.
%-------------------------
SIGMA = diag(sqrt(diag(D(1:R,1:R))));

%------------------
% Get elapsed time.
%------------------
% fprintf(1,'Elapsed time is %f.\n',toc);

%---------------------------------
% Project and display if required.
%---------------------------------
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
    proj = (ALPHA(:,1:R)')*myKernelMatrix(A,Pts,KTYPE,KPARAM);            

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
'VECT',A,...
'ALPHA',ALPHA,...
'SIGMA',SIGMA,...
'OPTIONS',OPTIONS,...
'PROJ',projcon...
);