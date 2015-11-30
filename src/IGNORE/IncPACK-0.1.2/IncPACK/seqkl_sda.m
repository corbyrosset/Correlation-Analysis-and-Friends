function [S,k,snaps,savesnaps,stats,passcon] =  ...
    seqkl_sda(S,k,A,                                    ...
               Utest,Stest,Vtest,                        ...
               reortho,display,debug,whch,               ...
               kstart,lmin,lmax,                         ...
               kmin,kmax,thresh,ttype,extrak,            ...
               snaps,savesnaps,stats,passes,numpasses,call);
% For implicit steepest descent passes, through A*[V_1 G_1 ...]

   global SEQKL_U SEQKL_V
   
   % get size of A
   [m,n] = size(A);

   muhat = 0;
   mubar = 0;
   flops = 0;

   % start clock, stopped below after call to seqkl_stat
   t0 = clock;

   % do we have enough passes left?
   % sda consumes three passes through A
   if (numpasses - passes) < 3,
      passcon = -1;
      if debug, fprintf('Not enough passes left for seqkl_sda.\n'); end
      return;
   end

   numsnaps = length(snaps);

   % allocate space for AX,X,Y,tau
   % ideally, we wouldn't have to allocate this over and over again. oh well.
   if (display > 1) || (debug > 0)
      fprintf(['*********************************** %8s ',...
               '**************************************\n'],'SD A');
   end
   if display > 1,
      fprintf('Constructing I+XY'' factorization for right basis...\n');
   end
   AX = zeros(m,2*k);
   X = zeros(n,2*k);
   Y = zeros(n,2*k);
   oldk = k;
   tau = zeros(2*k,1);


   if display > 1,
      fprintf('Computing gradient direciton...\n');
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % perform A'*U
   oldk = k;
   G = A'*SEQKL_U(:,1:k);
   flops = flops + 2*m*n*k;
   passes = passes + 1;

         function cbqr
         % QR Factorization of V, in place
            for j=1:2*k,
               [v,beta] = house(G(j:n,j));
               flops = flops + 2*(n-j);
               G(j:n,j:end) = G(j:n,j:end) - beta*v*(v'*G(j:n,j:end));
               flops = flops + 4*(n-j)*(2*k-j);
               tau(j) = beta;
               if j<n,
                  G(j+1:n,j) = v(2:n-j+1);
               end
            end
         end

         function blockhh
            Y(:,1) = [1;G(2:n,1)];
            X(:,1) = -tau(1)*Y(:,1);
            for j=2:2*k,
               v = [zeros(j-1,1);1;G(j+1:n,j)];
               flops = flops + 4*j*n;
               z = -tau(j)*(v + X(:,1:j-1)*(Y(:,1:j-1)'*v));
               X(:,j) = z;
               Y(:,j) = v;
            end
         end


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % use current V to produce W = I + X*Y'
   % compute householder vectors into V and scalars in tau
   G = [SEQKL_V(:,1:k),G];
   cbqr;
   % compute X,Y into WORKV1,WORKV2
   blockhh;
   % flops was updated dynamically


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % pass through A, incrementally computing A*X
   if display > 1,
      fprintf('Incrementally computing A*X...\n');
   end
   i = 1;
   while i<=n,
      % choose the maximum update size for best level 3
      lup = lmax;
      % now clamp it
      if i-1+lup > n
         lup = n-i+1;
      end
      ip  =  i+lup-1;
      
      AX(:,:) = AX(:,:) + A(:,i:ip)*X(i:ip,:);
      i = i+lup;
   end
   flops = flops + 2*m*n*size(X,2);
   passes = passes + 1;


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % compute incremental SVD of A*W = A + A*X*Y'
   i = 1;
   % start from scratch
   k = 0;
   while i<=n,
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % print header
      % but not the first time through, it was printed above
      if ((display > 1) || (debug > 0)) && i > 1,
         fprintf(['*********************************** %8s ',...
                  '**************************************\n'],'Imp SD A');
      end

      if i > 1,
         % start clock, stopped below after call to seqkl_stat
         t0 = clock;
      end
      
      if i==1 && k==0,

         % init with an SVD of the first kstart columns of AW
         k = kstart;
         ip = kstart;
         if debug, fprintf('DBG  Performing initial QR of first %d columns...\n',k); end
         SEQKL_U(:,1:k) = A(:,i:ip) + AX*Y(i:ip,:)';
         flops = flops + 2*m*k^2 + m*k;
         [SEQKL_U(:,1:k),R] = qr(SEQKL_U(:,1:k),0);
         flops = flops + 4*m*k^2;
         [Ur,Sr,Vr] = svd(triu(R));
         S(1:k) = diag(Sr);
         SEQKL_U(:,1:k) = SEQKL_U(:,1:k)*Ur;
         flops = flops + 2*m*k^2;
         SEQKL_V(1:k,1:k) = Vr;
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % update pointers: i,vi,vr,k
         i=k+1;
         vr = k;

      else

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % choose the (globally) optimal column size
         lup = clamp( lmin, lmax, round(k/sqrt(2)) );
         % now clamp it
         if i-1+lup > n,
            lup = n-i+1;
         end
         ip  =  i+lup-1;
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % printing
         if display > 1,
            fprintf('Expanding with columns %d through %d...\n',i,ip);
         end
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % compute/put update columns into appropriate part of U
         SEQKL_U(1:m,k+1:k+lup) = A(:,i:ip) + AX*Y(i:ip,:)';
         flops = flops + 2*m*oldk*lup;
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Update factorization
         [S,knew,fc] = seqkl_update(reortho,whch,m,n,i,k,lup,kmin,kmax,extrak,ttype,thresh,S,debug);
         flops = flops + fc;
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Keep track of muhat, mubar, discarded data
         if k+lup > knew
            muhat = max(muhat,S(knew+1));
            mubar = mubar + S(knew+1)^2;
         end
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % update pointers: i,vr,k
         k=knew;
         i=i+lup;

      end   % while i <= n

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Collect data
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      colsdone = n*passes+i-1;
      stat = seqkl_stat(SEQKL_U,S,                                ...
                         [SEQKL_V(1:i-1,1:k);zeros(n-i+1,k)]         ...
                           + X*(Y(1:i-1,:)'*SEQKL_V(1:i-1,1:k)),     ...
                         k,n,colsdone,muhat,mubar,flops,Utest,Stest,Vtest); 
      stat.time = etime(clock,t0);
      stat.call = call;
      if isempty(stats),
         stats = stat;
      else 
         stats(end+1) = stat;
      end

      % reset flops for the next update; this happens after every save to stats
      flops = 0;

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Print
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      if display > 0,
         fprintf(' rank: %*d   sum(sigma):%13.6e   columns: %*d/%*d   pass: %*d of %*d\n', ...
                 width(kmax), k, sum(S(1:k)), width(n), i-1, width(n), n, ...
                 width(numpasses), passes+1, width(numpasses), numpasses);
      end
      if display > 1,
         seqkl_disp2(stats(end));
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Save snapshots of factorization
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      while length(snaps) > 0,
         % save a snapshot if 
         % - we are on it or we have passed it, AND
         % - it is not equivalent to the previous snapshot
         % 
         % we assume that the snapshots are in order, so that we can quit
         % when we hit a snapshot larger than where we are
         colsdone = n*passes+i-1;
         if colsdone < snaps(1),
            break;
         elseif colsdone >= snaps(1),
            % don't save the same snapshot twice
            if length(savesnaps) == 0 || colsdone > savesnaps(end).i,
               cursnap.i = colsdone;
               cursnap.U(1:m,1:k) = SEQKL_U(1:m,1:k);
               cursnap.V(1:n,1:k) = [SEQKL_V(1:i-1,1:k);zeros(n-i+1,k)] + X*(Y(1:i-1,:)'*SEQKL_V(1:i-1,1:k));
               cursnap.S(1:k) = S(1:k);
               if isempty(savesnaps),
                  savesnaps = cursnap;
               else
                  savesnaps(end+1) = cursnap;
               end
               clear cursnap;
            end
            % remove this snapshot from the list
            snaps = snaps(2:end);
         end
      end

   end % while

   % rotate V by W
   SEQKL_V(1:n,1:k) = SEQKL_V(1:n,1:k) + X*(Y'*SEQKL_V(1:n,1:k));
   % add this to the last flop count
   stats(end).flops = stats(end).flops + n*k + 4*n*size(X,2)*k;

   passcon = 3;

end % function seqkl_sda


function [v,beta] = house(x)
   n = length(x);
   if n == 1
      v = 1;
      beta = 2;
      return;
   end
   sigma = x(2:n)'*x(2:n);
   v = [1;x(2:n)];
   if sigma == 0
      beta = 0;
   else
      mu = sqrt(x(1)*x(1)+sigma); 
      if x(1) <= 0
         v(1) = x(1) - mu;
      else
         v(1) = -sigma/(x(1)+mu);
      end
      beta = 2*v(1)*v(1)/(sigma+v(1)*v(1));
      v = v/v(1);
   end
end
