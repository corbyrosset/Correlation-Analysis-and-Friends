function [S,k,snaps,savesnaps,stats,passcon] =              ...
   seqkl_stdpass(S,k,A,                                    ...
                  Utest,Stest,Vtest,                        ...
                  reortho,display,debug,whch,               ...
                  kstart,lmin,lmax,                         ...
                  kmin,kmax,thresh,ttype,extrak,            ...
                  snaps,savesnaps,stats,passes,numpasses,call);
% For 'std/echo', 'error-based refine' or 'error-based expand'

   % get size of A
   [m,n] = size(A);

   global SEQKL_U SEQKL_V

   % do we have enough passes left?
   % stdpass consumes one pass through A
   if (numpasses - passes) < 1,
      passcon = -1;
      if debug, fprintf('Not enough passes left for seqkl_stdpass.\n'); end
      return;
   end
   passcon = 1;

   muhat = 0;
   mubar = 0;
   flops = 0;

   % start clock, stopped below after call to seqkl_stat
   t0 = clock;

   i = 1;
   while i<=n,

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % print header
      if (display > 1) || (debug > 0)
         fprintf(['*********************************** %8s ',...
                  '**************************************\n'],'stdpass');
      end

      if i > 1,
         % start clock, stopped below after call to seqkl_stat
         t0 = clock;
      end

      if i==1 && k==0,
         
         % init with an SVD of the first kstart columns of A
         ip = k;
         k = kstart;
         if debug, fprintf('DBG  Performing initial QR of first %d columns...\n',k); end
         [SEQKL_U(:,1:k),R] = qr(A(:,1:k),0);
         flops = flops + 4*m*k^2;
         [Ur,Sr,Vr] = svd(triu(R));
         S(1:k) = diag(Sr);
         SEQKL_U(:,1:k) = SEQKL_U(:,1:k)*Ur;
         flops = flops + 2*m*k^2;
         SEQKL_V(1:k,1:k) = Vr;
         % sort singular values
         if (isequal(whch,'S')) 
            [S(1:k),order] = sort(S(1:k),1,'ascend');
         else
            [S(1:k),order] = sort(S(1:k),1,'descend');
         end
         SEQKL_U(:,1:k)   = SEQKL_U(:,order);
         SEQKL_V(1:k,1:k) = SEQKL_V(1:k,order);
         % start partway through the first pass
         i=k+1;

      else

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % choose the (globally) optimal column size
         lup = clamp( lmin, lmax, round(k/sqrt(2)) );
         % now clamp it
         if i-1+lup > n
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
         SEQKL_U(1:m,k+1:k+lup) = A(:,i:ip);
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Update factorization
         [S,knew,fc] = seqkl_update(reortho,whch,m,n,i,k,lup,kmin,kmax,extrak,ttype,thresh,S,debug);
         flops = flops + fc;
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Keep track of muhat, mubar, discarded data
         if k+lup > knew
            muhat = max(muhat,max(S(knew+1:k+lup)));
            mubar = mubar + max(S(knew+1:k+lup))^2;
         end
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % update pointers: i,k
         k=knew;
         i=i+lup;
      end

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Collect data
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % test right basis
      colsdone = n*passes+i-1;
      stat = seqkl_stat(SEQKL_U,S,SEQKL_V,k,i-1,colsdone,muhat,mubar,flops,Utest,Stest,Vtest); 
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
                 width(kmax), k, sum(S(1:k)), width(n), stats(end).X, width(n), n, ...
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
               cursnap.V(1:i-1,1:k) = SEQKL_V(1:i-1,1:k);
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

end % function
