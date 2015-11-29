function [S,k,snaps,savesnaps,stats,passcon] =  ...
    seqkl_sdb(S,k,A,                                 ...
               Utest,Stest,Vtest,                        ...
               reortho,display,debug,whch,               ...
               kmin,kmax,thresh,ttype,extrak,            ...
               snaps,savesnaps,stats,passes,numpasses,call);
% For implicit steepest descent passes, through A*[V_1 G_1]

   global SEQKL_U SEQKL_V
   
   % get size of A
   [m,n] = size(A);

   muhat = 0;
   mubar = 0;
   flops = 0;

   % start clock, stopped below after call to seqkl_stat
   t0 = clock;

   % do we have enough passes left?
   % sdb consumes two passes through A
   if (numpasses - passes) < 2,
      passcon = -1;
      if debug, fprintf('Not enough passes left for seqkl_sdb.\n'); end
      return;
   end

   numsnaps = length(snaps);

   % allocate space for AX,X,Y,tau
   % ideally, we wouldn't have to allocate this over and over again. oh well.
   if (display > 1) || (debug > 0)
      fprintf(['*********************************** %8s ',...
               '**************************************\n'],'SD B');
   end
   if display > 1,
      fprintf('Computing gradient direciton...\n');
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % perform A'*U
   oldk = k;
   AtU = A'*SEQKL_U(:,1:k);
   flops = flops + 2*m*n*k;
   passes = passes + 1;

   % subtract A^T U - V S                 (nk)
   G = AtU - SEQKL_V(1:n,1:k)*diag(S(1:k));
   flops = flops + n*k;

   % qr factorization W2 R2 = A^T U - V S (4nk2)
   G = [SEQKL_V,G];
   [G,dummy] = qr(G,0);
   G = G(:,k+1:end);
   flops = flops + 4*n*k^2;

   % perform partial IncSVD of A*W
   %   - first step is implicit
   %   - second step is explicit

   % IncSVD update U S | A W2              (12mk2)
   % init with an SVD of the first kstart columns of AW
   oldV = SEQKL_V(1:n,1:k);
   SEQKL_V(1:k,1:k) = eye(k);
   i   = k+1;
   ip  = 2*k;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % printing
   if display > 1,
      fprintf('Expanding with columns %d through %d...\n',i,ip);
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % compute/put update columns into appropriate part of U
   % multiply A W2                         (2mnk, one pass)
   SEQKL_U(1:m,k+1:2*k) = A*G;
   flops = flops + 2*m*n*k;
   passes = passes + 1;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Update factorization
   [S,k,fc] = seqkl_update(whch,reortho,m,n,i,k,k,kmin,kmax,extrak,ttype,thresh,S,debug);
   flops = flops + fc;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Keep track of muhat, mubar, discarded data
   if 2*oldk > k
      muhat = max(muhat,S(k+1));
      mubar = mubar + S(k+1)^2;
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % rotate V by W
   SEQKL_V(1:n,1:k) = oldV*SEQKL_V(1:oldk,1:k) + G*SEQKL_V(oldk+1:2*oldk,1:k);
   flops = flops + 2*n*oldk*k;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Collect data
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   colsdone = n*passes;
   stat = seqkl_stat(SEQKL_U,S,        ...
                      SEQKL_V,          ...
                      k,n,colsdone,muhat,mubar,flops,Utest,Stest,Vtest); 
   stat.time = etime(clock,t0);
   stat.call = call;
   if isempty(stats),
      stats = stat;
   else 
      stats(end+1) = stat;
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Print
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if display > 0,
      fprintf(' rank: %*d   sum(sigma):%13.6e   columns: %*s/%*s   pass: %*d of %*d\n', ...
              width(kmax), k, sum(S(1:k)), width(n), '', width(n), '', ...
              width(numpasses), passes, width(numpasses), numpasses);
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
      colsdone = n*passes;
      if colsdone < snaps(1),
         break;
      elseif colsdone >= snaps(1),
         % don't save the same snapshot twice
         if length(savesnaps) == 0 || colsdone > savesnaps(end).i,
            cursnap.i = colsdone;
            cursnap.U(1:m,1:k) = SEQKL_U(1:m,1:k);
            cursnap.V(1:n,1:k) = SEQKL_V(1:n,1:k);
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

   passcon = 2;

end % function seqkl_impsd
