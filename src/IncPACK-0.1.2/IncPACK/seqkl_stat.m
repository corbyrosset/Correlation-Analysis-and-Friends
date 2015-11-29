function stat = seqkl_stat(U,S,V,k,vr,colsdone,muhat,mubar,flops,Utest,Stest,Vtest);

   % loss of orthogonality and conditioning
   stat.flops = flops;
   stat.X = colsdone;
   stat.orthu = norm(eye(k) - U(:,1:k)'*U(:,1:k),'fro');
   stat.orthv = norm(eye(k) - V(1:vr,1:k)'*V(1:vr,1:k),'fro');
   if S(k) == 0
      c = Inf;
   else
      c = S(1) / S(k);
   end
   stat.cond = c;
   stat.estorthu = c*c * 2 * (2*eps);
   % stat.recnorm = nan;
   stat.muhat = muhat;
   stat.mubar = mubar;
   stat.sumsigma = sum(S(1:k));
   % predicted subspace error
   % stat.estPu = atan(muhat^2 / (S(k)^2 - muhat^2));
   % stat.estPv = atan(2*muhat*S(1) / (S(k)^2 - muhat^2));
   % distance from test
   if ~isempty(Utest),
      UtU = Utest'*U(:,1:k);
      stat.cdistu = norm(acos(svd(UtU)));
      stat.sdistu = norm(asin(svd(U(:,1:k) - Utest*UtU)));
   else
      stat.cdistu = nan;
      stat.sdistu = nan;
   end
   if ~isempty(Vtest) && vr == size(Vtest,1),
      VtV = Vtest'*V(1:vr,1:k);
      stat.cdistv = norm(acos(svd(VtV)));
      stat.sdistv = norm(asin(svd(V(1:vr,1:k) - Vtest*VtV)));
   else
      stat.cdistv = nan;
      stat.sdistv = nan;
   end
   if ~isempty(Stest),
      lk = min(k,length(Stest));
      stat.dists = norm(S(1:lk) - Stest(1:lk));
   else
      stat.dists = nan;
   end
