function [ Z, BETA ] = SimCompress( A, ALPHA, ZNUM, KTYPE, KPARAM )
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

% Multi-term RBF approximations.
if KTYPE == 2
    for p = 1:1:( size(ALPHA,2)*ZNUM )

        % Selection of the starting point out of the current library.
        % The point in which is the objective function minimal is taken.
        % Minimum over 50 randomly drawn points is used.
        %--------------------------------------------------------------
        if p==1        
            tarVect = A;
            tarCoff = ALPHA;
        else
            tarVect = [ A Z ];
            tarCoff = [ ALPHA ; -BETA ];
        end        
        rand_inx = randperm( size(tarVect,2) );
        rand_inx = rand_inx(1:min(size(tarVect,2),50));
        potZ = tarVect(:,rand_inx);                    
        fval = myGaussianKernelMatrix(potZ,tarVect,KPARAM)*tarCoff;        
        fval = -fval.^2;
        [dummy, inx ] = min(min( fval ));
        z = potZ(:,inx ); % Starting point.
        
        %-----------------------
        % Fixed-point iteration.
        %-----------------------                
        % % Scholkopf.
        % %-----------        
        % % For numerator.
        % accum = zeros( size(z,2), 1);
        % for j=1:size( tarVect, 2 )
        %     accum = accum + myGaussianKernelMatrix(tarVect(:,j),z,KPARAM)*tarVect(:,j);
        % end
        % znew = zeros( size(z,1), 1 );
        % for k=1:size( tarCoff, 2 )
        %     accum2 = zeros( size(z,2), 1 );
        %     for i=1:size( tarVect, 2 )                
        %         accum2 = accum2 + tarCoff(i,k)*myGaussianKernelMatrix(tarVect(:,i),z,KPARAM)*accum;
        %     end
        %     znew = znew + accum2;
        % end
        % 
        % % For denominator.
        % accum3 = 0;
        % for j=1:size( tarVect, 2 )
        %     accum3 = accum3 + myGaussianKernelMatrix(tarVect(:,j),z,KPARAM);
        % end
        % denom = 0;
        % for k=1:size( tarCoff, 2 )
        %     accum4 = 0;
        %     for i=1:size( tarVect, 2 )                
        %         accum4 = accum4 + tarCoff(i,k)*myGaussianKernelMatrix(tarVect(:,i),z,KPARAM)*accum3;
        %     end
        %     denom = denom + accum4;
        % end
        % 
        % znew = znew./denom;        
        % %-----------
        
        % % Scholkopf matrix form.
        % %-----------------------
        % change = 1;
        % if (change > 1e-6)        
        %     intres = myGaussianKernelMatrix(tarVect,z,KPARAM);
        %     numer = sum((tarVect*intres)*((tarCoff'*intres)'),2);
        %     denom = sum(intres)*sum( tarCoff'*intres );
        %     znew = numer./denom;
        %     change = sum((z-znew).^2);
        %     z = znew;
        % end
        
        % % Chin.
        % %------
        % sumk = (tarCoff')*myGaussianKernelMatrix(tarVect,z,KPARAM);
        % % For numerator.        
        % accum2 = zeros( size(tarVect,1), 1 );
        % for k=1:size( tarCoff, 2 )            
        %     accum = zeros( size(tarVect,1), 1);
        %     for j=1:size( tarVect, 2 )
        %         accum = accum + tarVect(:,j).*tarCoff(j,k)*myGaussianKernelMatrix( tarVect(:,j), z, KPARAM );
        %     end
        %     accum2 = accum2 + sumk(k).*accum;
        % end
        % % For denominator.
        % accum4 = 0;
        % for k=1:size( tarCoff, 2 )            
        %     accum3 = 0;
        %     for j=1:size( tarVect, 2 )
        %         accum3 = accum3 + tarCoff(j,k)*myGaussianKernelMatrix( tarVect(:,j), z, KPARAM );
        %     end
        %     accum4 = accum4 + sumk(k)*accum3;
        % end        
        % znew = accum2./accum4;
        
        % Chin matrix form.
        %------------------
        change = 1;
        if (change > 1e-10)                
            sumk = (tarCoff')*myGaussianKernelMatrix(tarVect,z,KPARAM);
            intres = myGaussianKernelMatrix(tarVect,z,KPARAM);
            numer = zeros( size(tarVect,1), 1);
            for k=1:size( tarCoff, 2 )
                numer = numer + (tarVect*( tarCoff(:,k).*intres ) ).*sumk(k);
            end
            denom = 0;
            for k=1:size( tarCoff, 2 )
                denom = denom + sumk(k).*sumk(k);
            end
            znew = numer./denom;
            change = sum((z-znew).^2);
            z = znew;
        end
        
        %---------------------
        % Store new pre-image.
        %---------------------
        if p==1
            Z = z;
        else
            Z = [ Z z ];
        end

        %------------------------
        % Coefficient estimation.
        %------------------------
        Kz = myGaussianKernelMatrix(Z,Z,KTYPE);
        Kzs = myGaussianKernelMatrix(Z,A,KTYPE);
        if (rcond(Kz) < 1e-6)
            BETA = pinv(Kz)*Kzs*ALPHA;
        elseif (isnan(rcond(Kz)) == 1)
            fprintf(1,'Condition number is NaN. ');
            Z = NaN;
            BETA = NaN;
            return;
        else      
            BETA = inv(Kz)*Kzs*ALPHA;
        end
        
        %-------------------------
        % Get angular differences.
        %-------------------------
        % P = ALPHA'*myGaussianKernelMatrix(A,Z,KPARAM)*BETA;
        % [ u s v ] = svd(P);
        % fprintf(1,'Chordal dist = %f.\n',norm(acos(s),'fro')./sqrt(2));
        
    end
else
    error('Only RBF kernels are supported.\n');
end

% % Get all RS expansions.
% for r = 1:nALPHA    
%     %-------------------------------------
%     % The following uses the STPR toolbox.
%     %-------------------------------------
%     if (KTYPE == 2)
%         
%         model.Alpha = ALPHA(:,r);
%         model.sv.X = A;
%         if r>1
%             model.InitPreimg = Z;
%         end
%         model.options.ker = 'rbf';
%         model.options.arg = KPARAM;
%         options.nsv = ZNUM;
%         %options.preimage = 'rbfpreimg2';
%         options.preimage = 'MyRbfpreimg_opt';
%         %options.preimage = 'MyRbfpreimg_fpi';
%         options.verb = 0;
%         red_model = MyRsrbf(model,options);
%         if (isnan(red_model.Alpha) == 1)            
%             Z = NaN;
%             BETA = NaN;
%             return;
%         end
%         if r==1
%             Z = red_model.sv.X;
%             BETA = red_model.Alpha;
%         else
%             Z = red_model.sv.X;
%             BETA = [ [ BETA ; zeros( size(red_model.Alpha,1)-size(BETA,1),r-1) ] red_model.Alpha ];
%         end
%         
%     elseif ((KTYPE == 1)&&(KPARAM == 2))        
%         
%         model.Alpha = ALPHA(:,r);
%         model.b = 0;
%         model.sv.X = A;
%         model.nsv = nA;
%         model.options.ker = 'poly';
%         model.options.arg = [2 0];
%         red_model_1 = rspoly2(model,ZNUM);
%         
%         if r==1            
%             Z = red_model_1.sv.X;
%             BETA = red_model_1.Alpha;            
%         else
%             Z = [ Z red_model_1.sv.X ];
%             BETA = [ [ BETA ; zeros( size(red_model_1.Alpha,1),r-1) ] [ zeros(size(BETA,1),1) ; red_model_1.Alpha ] ];
%         end
%     else
%         error('Only RBF kernels and 2nd-degree polynomial kernal are supported!');
%         return;        
%     end
% end