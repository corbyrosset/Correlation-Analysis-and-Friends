function varargout = seqkl(varargin)
% SEQKL    Dynamic Rank Sequential Karhunen-Loeve
%
% Description:
%  This function implements the dynamic-rank Sequential Karhunen-Loeve,
%  making one pass to approximate the dominant SVD U*S*V' and (optionally)
%  later passes improve on it.
%
% Synopsis:
%  S = SEQKL(A) returns a vector S with approximations for the largest 6 singular 
%  values of the matrix A. The elements of S are guaranteed to be non-negative 
%  and sorted ascending.
% 
%  [U,S] = SEQKL(A) returns a matrix U with orthonormal columns approximating
%  the left singular vectors corresponding to the approximate singular values in 
%  S. S is a diagonal matrix with non-negative and non-decreasing elements.
%
%  [U,S,V] = SEQKL(A) returns a matrix V with orthonormal columns approximating
%  the right singular vectors.
% 
%  [U,S,V,O] = SEQKL(A) returns a structure O with statstics from the computation
%  of U, S and V.
% 
%  [...] = SEQKL(A,K) computes approximations for the largest K singular values
%  of A.
% 
%  [...] = SEQKL(A,KMAX,THRESH) computes approximations for up to KMAX of the
%  largest singular values satisfying the relative threshhold THRESH, as follows:
%            If OPTS.ttype == 'rel', preserve all singular values such that
%               sigma >= thresh*min(sigma)    if whch == 'L'
%               sigma <= thresh*min(sigma)    if whch == 'S'
%            If OPTS.ttype == 'abs', preserve all singular values such that
%               sigma >= thresh               if whch == 'L'
%               sigma <= thresh               if whch == 'S'
% 
%  [...] = SEQKL(A,K,...,'S') computes approximations for the smallest 
%  singular values.
%
%  [...] = SEQKL(A,K,...,OPTS) allows the specification of other options, as
%  listed below.
%   
% Optional input:
%   opts.numpasses   - number of times to pass through A
%   opts.mode        - Update mode, default='auto2'
%        'restart'   - compute SVD of A*W, W=[V Vperp], using last V
%        'sda'       - IncSVD-based, implicit steepest descent on A'*A, variant A
%        'sdb'       - IncSVD-based, implicit steepest descent on A'*A, variant B
%   opts.thresh      - Tolerance for determining rank at each step. See above.
%   opts.ttype       - Rank threshhold type: 'abs' or 'rel'. See above. 
%                      Default='rel'
%   opts.reortho     - perform two-steps of classical Gram-Schmidt during SVD update
%                      Default='yes'
%   opts.snaps       - array of snapshot requests, by number columns processed
%   opts.kstart      - number of columns used to initial decomp (lmax)
%   opts.kmin        - minimum rank tracked by method, default=1
%   opts.extrak      - extra dimension tracked.
%                      applied after thresh-chosen k, before accounting for kmax.
%   opts.finalttype  - final pass: same as opts.ttype, default=opts.ttype
%   opts.finalthresh - final pass: same as opts.fthresh, default=opts.fthresh
%   opts.finalkmin   - final pass: same as opts.kmin, default=opts.kmin
%   opts.finalkmax   - final pass: same as opts.kmax, default=kmax
%   opts.lmin        - minimum value for l at each step, default=1
%   opts.lmax        - maximum value for l at each step, default=kmax/sqrt(2)
%   opts.disp        - print while running or not, default=0 
%           0  -  silent
%           1  -  one-liners
%           2  -  chatty
%   opts.debug: debugging checks and output [{0} | 1]. Expensive!!!
%   opts.paramtest: parameter test          [{0} | 1]
%           do not actually run the algorithm, but instead test parameters,
%           set the defaults, and return them in a struct
%           Example: params = seqkl(A,kmax,thresh,opts);
%   opts.[USV]    - initial factorization for starting multipass method:
%          .U    - initial left vectors    (m by o) 
%          .S    - initial left vectors    (o by 1) 
%          .V    - initial left vectors    (n by o)
%
% Example:
%  data = randn(1000,100);
%  [U,S,V,OPS] = seqkl(data,10,.20,'S');
%
% See also EIG, SVD, RSVD
%

% About: IncPACK - The Incremental SVD Package
% Version 0.1.2
% (C) 2001-2012, Written by Christopher Baker
% <a href="http://www.fsu.edu">The Florida State University</a>
% <a href="http://www.scs.fsu.edu">School of Computational Science</a>

% TODO
% * stop using globals for U and V. switch to MATLAB handle classes.
% * add back error estimates

% Modifications:
% 18-dec-2012, CGB (added modified BSD license)
% 11-apr-2008, CGB (switched order of inputs: thresh before whch. improved documentation.)
%  8-apr-2008, CGB (got rid of recovery mode, simplified other modes, switched from global
%                   data to handle class)
%  7-jul-2006, CGB (added SDB,SDA; fixed flop counting; renamed auto modes)
% 16-feb-2006, CGB (redid input processing, restructured, rewrote multipass code, 
%                   updated printing, misc)
% 29-jan-2006, CGB 
% 14-sep-2005, CGB (added option to compute dominated SVD)
% 04-aug-2004, CGB (added Gu/GV-based theta/pi estimation)
% 28-oct-2003, CGB

% don't print file/lineno on warnings
warning off backtrace;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we accept at most three output args: U, S, V and Ops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargout > 4)
    error('Too many output arguments. See ''help seqkl''')
end
if length(varargin) < 1,
    error('Not enough input arguments. See ''help seqkl'' for more.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% grab input arguments: A,kmax,thresh,opts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A,kmax,whch,thresh,opts] = get_args(varargin{:});
[m,n] = size(A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% strings
ttype = 'rel';
mode = 'restart';
fttype = [];
reortho = 'yes';
% ints
verbosity = 0;
kmin = 1;
lmin = 1;
lmax = round(kmax/sqrt(2));
kstart = lmax;
fkmin = 1;
fkmax = kmax;
extrak = 0;
numpasses  = 1;
paramtest   = 0;
debug = 0;
earlystop = n;
% floats
fthresh = [];
% arrays
snaps = [];
Utest = [];
Vtest = [];
Stest = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get parameters from opts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% strings
[mode,opts] = get_string(opts,'mode','Update mode',mode,...
                           {'restart','sda','sdb'});
[whch,opts] = get_string(opts,'which','Target singular values',whch,...
                           {'L','S'});
[ttype,opts] = get_string(opts,'ttype','Threshhold type',ttype,...
                           {'abs','rel'});
[fttype,opts] = get_string(opts,'finalttype','Final threshhold type',ttype,...  % default is ttype
                           {'abs','rel'});
[reortho,opts] = get_string(opts,'reortho','Reorthogonalization flag',reortho,...
                           {'yes','no'});
% ints
[verbosity,opts] = get_int(opts,'verbosity','Verbosity level',verbosity);
[kmin,opts] = get_int(opts,'kmin','Minimum rank',kmin,1,kmax);
[lmin,opts] = get_int(opts,'lmin','Minimum update size',lmin,1);     % do lmin first so it takes precedent
[lmax,opts] = get_int(opts,'lmax','Maximum update size',lmax,lmin);  % bound lmax by lmin
[kstart,opts] = get_int(opts,'kstart','Starting rank',kstart,1);
[fkmax,opts] = get_int(opts,'finalkmax','Final maximum rank',fkmax,1,kmax);   % do fkmax first so it takes precedent
[fkmin,opts] = get_int(opts,'finalkmin','Final minimum rank',fkmin,1,fkmax);   % bound fkmin by fkmax
[extrak,opts] = get_int(opts,'extrak','Extra rank modifier',extrak,1);
[numpasses,opts] = get_int(opts,'numpasses','Number of passes through A',numpasses,1);
[paramtest,opts] = get_int(opts,'paramtest','Parameter test flag',paramtest);
[debug,opts] = get_int(opts,'debug','Debug flag',debug);
[earlystop,opts] = get_int(opts,'earlystop','Early rotpass stop',earlystop);
% scalar floats
if isnan(thresh), % not specified as arg to seqkl(...)
   % these defaults only make sense if ttype == 'rel'
   switch([ttype whch])
   case {'absS','relS'}
      thresh = inf;
   case {'absL','relL'}
      thresh = 0;
   end
   [thresh,opts] = get_float(opts,'thresh','Threshhold',thresh,0,inf,[1 1]);
end
[fthresh,opts] = get_float(opts,'finalthresh','Final threshhold',thresh,0,inf,[1 1]);  % default is thresh
% arrays
[snaps,opts] = get_float(opts,'snaps','Snapshot indices',snaps);
if ~isempty(snaps),
   if isvector(snaps),
      snaps = sort(snaps);
   else
      warning(['opts.snaps must be a vector']);
      snaps = [];
   end
end
[Utest,opts] = get_float(opts,'Utest','Comparison U',Utest,0,0,[m kmax]);
[Vtest,opts] = get_float(opts,'Vtest','Comparison V',Vtest,0,0,[n kmax]);
[Stest,opts] = get_float(opts,'Stest','Comparison S',Stest);
if ~isempty(Stest) && ~isvector(Stest),
   Stest = diag(Stest);
   if (isequal(whch,'L'))
      Stest = sort(Stest,1,'descend');
   else
      Stest = sort(Stest,1,'ascend');
   end
end
% init is extra special
if isfield(opts,'U') && size(opts.U,1) == m && size(opts.U,2) > 0 && ...
   isfield(opts,'V') && size(opts.V,1) == n && size(opts.V,2) > 0 && ...
   isfield(opts,'S') && length(opts.S) > 0,
   hasInit = 1;
   % this copy is cheap; matlab should use a reference, and not allocate the space
   initU = opts.U;
   initS = opts.S;
   if ~isvector(initS),
      initS = diag(initS);
   end
   initV = opts.V;
   opts = rmfield(opts,{'U','S','V'});
else
   hasInit = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If performing a parameter test, save params and exit now
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if paramtest,
   theparams.whch       = whch;
   theparams.kmin       = kmin;
   theparams.kmax       = kmax;
   theparams.kstart     = kstart;
   theparams.extrak     = extrak;
   theparams.thresh     = thresh;
   theparams.ttype      = ttype;
   theparams.mode       = mode;
   theparams.reortho    = reortho;
   theparams.verbosity    = verbosity;
   theparams.lmin       = lmin;
   theparams.lmax       = lmax;
   theparams.finalttype  = fttype;
   theparams.finalthresh = fthresh;
   theparams.finalkmin  = fkmin;
   theparams.finalkmax  = fkmax;
   theparams.snaps      = snaps;
   theparams.numpasses  = numpasses;
   theparams.hasInit    = hasInit;
   theparams.debug      = debug;
   
   varargout{1} = theparams;
   return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allocate large data structures first, as early as possible
% These are stored globally, so that they can efficiently be modified 
% across routines.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global SEQKL_U SEQKL_V
maxudim = max(kmax,kstart)+lmax;
maxvdim = max(kmax,kstart);
if debug, fprintf('DBG  Allocating space for bases\n'); end
SEQKL_U = zeros(m,maxudim);
SEQKL_V = zeros(n,maxvdim);
S = zeros(maxudim,1);
clear maxudim maxvdim;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Any fields in OPTS that were not consumed should be passed
% on to OPS. Init OPS now.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OPS = opts;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize misc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numsnaps = length(snaps);
savesnaps = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial factorization and Initialize variables
%
% Pointers/counters
% i          points to which column of A we use to update
% vr         denotes the number of valid rows in V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 0;
if hasInit,
   k = min(min(size(initU,2),size(initV,2)),length(initS));
   k = min(k,kmax);
end
if k > 0,
   if debug, fprintf('DBG  Initializing from input factorization\n'); end
   SEQKL_U(:,1:k)   = initU(:,1:k);
   SEQKL_V(1:n,1:k) = initV(:,1:k);
   S(1:k)     = initS(1:k);
   % vr tells us the number of valid rows in V. Used for measuring orthonormality,saving snapshots
   vr = n;
else
   vr = 0;
end
snapdone = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Record initial data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if k > 0,
   stats = seqkl_stat(SEQKL_U,S,SEQKL_V,k,vr,0,0,0,0,Utest,Stest,Vtest);
else
   stats = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save snapshots (only if we currently have a factorization)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if k > 0,
   for snpc=1:numsnaps,
      % save a snapshot if 
      % - we are on it
      % - we have passed it and it is not saved (in which case, snapdone++)
      % 
      % we assume that the snapshots are in order, so that we can quit
      % when we hit a snapshot larger than where we are
      if 0 < snaps(snpc),
         break;
      elseif 0 == snaps(snpc),
         savesnaps{snpc}.i = 0;
         savesnaps{snpc}.U(1:m,1:k) = SEQKL_U(1:m,1:k);
         savesnaps{snpc}.V(1:m,1:k) = SEQKL_V(1:vr,1:k);
         savesnaps{snpc}.S(1:k) = S(1:k);
      end
   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial printing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if k > 0,
   if (verbosity > 1) || (debug > 0)
      fprintf(['*********************************** %8s ',...
               '**************************************\n'],'Init');
   end
   if verbosity > 0,
      fprintf(' rank: %2.2d   sum(sigma):%13.6e   done: ____/____   pass: %d of %d\n', ...
              k, sum(S(1:k)), 0, numpasses);
   end
   if verbosity > 1,
      seqkl_disp2(stats(end));
   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outer Loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no passes yet
passes = 0;
call = 0;
while passes < numpasses,

   call = call + 1;
   if debug, fprintf('DBG  Passes left: %d\n',numpasses-passes); end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % We may need to
   % - change modes
   % - update pointers
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %
   % time to change modes? 
   % - if we had an initial factorization
   % - we are not on our first pass
   if k > 0,
      curmode = mode;
      if debug, fprintf('DBG  Switching to mode ''%s''\n',curmode); end
   else
      curmode = 'expand';
      if debug, fprintf('DBG  Setting mode ''expand''\n'); end
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Update factorization with new pass(es)
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if isequal(curmode,'expand'),
      % Update factorization with a new pass
      [S,k,snaps,savesnaps,stats,passcon] = seqkl_stdpass(                        ...
                                         S,k,A,                                    ...
                                         Utest,Stest,Vtest,                        ...
                                         reortho,verbosity,debug,whch,               ...
                                         kstart,lmin,lmax,                         ...
                                         kmin,kmax,thresh,ttype,extrak,            ...
                                         snaps,savesnaps,stats,passes,numpasses,call);
   elseif isequal(curmode,'restart'),
      % Perform restarted pass: A*[V ...]
      [S,k,snaps,savesnaps,stats,passcon] = seqkl_restart(                        ...
                                         S,k,A,                                    ...
                                         Utest,Stest,Vtest,                        ...
                                         reortho,verbosity,debug,whch,               ...
                                         kstart,lmin,lmax,                         ...
                                         kmin,kmax,thresh,ttype,extrak,            ...
                                         snaps,savesnaps,stats,passes,numpasses,call);
   elseif isequal(curmode,'sda'),
      % Perform impsd pass: A*[V Wgrad ...]
      [S,k,snaps,savesnaps,stats,passcon] = seqkl_sda(                            ...
                                         S,k,A,                                    ...
                                         Utest,Stest,Vtest,                        ...
                                         reortho,verbosity,debug,whch,               ...
                                         kstart,lmin,lmax,                         ...
                                         kmin,kmax,thresh,ttype,extrak,            ...
                                         snaps,savesnaps,stats,passes,numpasses,call);
   elseif isequal(curmode,'sdb'),
      % Perform impsd pass: A*[V Wgrad]
      [S,k,snaps,savesnaps,stats,passcon] = seqkl_sdb(                            ...
                                         S,k,A,                                    ...
                                         Utest,Stest,Vtest,                        ...
                                         reortho,verbosity,debug,whch,               ...
                                         kmin,kmax,thresh,ttype,extrak,            ...
                                         snaps,savesnaps,stats,passes,numpasses,call);
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if passcon == -1,
      break;
   else
      passes = passes + passcon;
   end

end   % while passes < numpasses


%%%%%%%%%%%%%%%%%%
% compute final k
oldk = k;
S = S(1:k);
if isequal(fttype,'abs'),
   % absolute threshold
   if isequal(whch,'L')
      k = sum( S >= fthresh );
   else
      k = sum( S <= fthresh );
   end
elseif isequal(fttype,'rel'),
   % relative threshold
   if isequal(whch,'L')
      k = sum( S >= fthresh*S(1) );
   else
      k = sum( S <= fthresh*S(1) );
   end
end
k = clamp(kmin,kmax,k);
if k < oldk,
   muhat = max(stats(end).muhat,S(k+1));
   mubar = stats(end).mubar + S(k+1)^2;
   % Collect data
   stat = seqkl_stat(SEQKL_U,S,SEQKL_V,k,n,0,muhat,mubar,0,Utest,Stest,Vtest);
   stat.time = 0;
   stat.call = stats(end).call;
   stats(end+1) = stat;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Print
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if verbosity > 1 || (debug > 0),
      fprintf(['*********************************** %8s ',...
               '**************************************\n'],'Final K');
   end
   if verbosity > 0,
      fprintf(' rank: %*d   sum(sigma):%13.6e\n', width(kmax), k, sum(S(1:k)));
   end
   if verbosity > 1,
      seqkl_disp2(stats(end));
   end
end


%%%%%%%%%%%%%%%%%%
% collect results
U = SEQKL_U(1:m,1:k);
S = S(1:k);
V = SEQKL_V(1:n,1:k);
clear global SEQKL_U SEQKL_V
% save timing, flop estimate, stats, snapshots
OPS.total_flops = sum([stats.flops]);
OPS.total_time = sum([stats.time]);
OPS.snaps = savesnaps;
OPS.stats = stats;

% print footer
if (verbosity > 1) || (debug > 0)
    fprintf(['***************************************', ...
             '********************************************\n']);
end
% print timing
if (verbosity > 0) || (debug > 0)
    fprintf('Total time is %f\n',OPS.total_time);
end

% send out the data
if nargout <= 1,
   varargout{1} = S;
else
   varargout{1} = U;
   varargout{2} = diag(S);
   if nargout >= 3, varargout{3} = V;   end
   if nargout >= 4, varargout{4} = OPS; end
end

end   % end function seqkl




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_string %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_string(opts,argname,argdesc,def,options)

    % Process inputs and do error-checking 
    errstr = sprintf('%s opts.%s must be: \n',argdesc,argname);
    errstr = [errstr, sprintf('%s  ',options{:})];
    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        if isstr(ret),
            for i = 1:length(options),
                if isequal(ret,options{i}),
                    valid = 1;
                    break;
                end
            end
        end
        if ~valid,
            error(errstr);
        end
         
        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_string




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_int %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_int(opts,argname,argdesc,def,lb,ub)

    if nargin < 6
        ub = inf;
        if nargin < 5,
            lb = -inf;
        end
    end

    % Process inputs and do error-checking 
    errstr = sprintf('%s opts.%s must be an integer in [%d,%d]',...
                     argdesc,argname,lb,ub);
    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        % check that it is an int
        if isnumeric(ret),
            ret = floor(ret);
            % check size (1 by 1) and bounds
            if isequal(size(ret),[1 1]) && lb <= ret && ret <= ub,
                valid = 1;
            end
        end
        if ~valid,
            error(errstr);
        end

        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_int




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_float %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_float(opts,argname,argdesc,def,lb,ub,sz)
% only test bounds if sz == [1 1]

    if nargin < 7,
        sz = [];
    end
    if nargin < 6
        ub = inf;
    end
    if nargin < 5,
        lb = -inf;
    end

    % Process inputs and do error-checking 
    if isequal(sz,[1 1]),
        errstr = sprintf('%s opts.%s must be a scalar in [%d,%d]',...
                         argdesc,argname,lb,ub);
    elseif ~isempty(sz),
        errstr = sprintf('%s opts.%s must be an array of dimension %d by %d',...
                         argdesc,argname,sz(1),sz(2));
    % else, there are no tests, and no possible failure
    end

    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        % check that it is an int
        if isnumeric(ret),
            ret = double(ret);
            % no size request, no checks at all
            if isempty(sz),
                valid = 1;
            % if scalar requested, perform bounds check
            elseif isequal(sz,[1 1]),
                if isequal(sz,size(ret)) && lb <= ret && ret <= ub,
                    valid = 1;
                end
            % if matrix requested, just check size
            elseif isequal(sz,size(ret)),
                valid = 1;
            end
        end
        if ~valid,
            error(errstr);
        end

        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_float




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_args   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,kmax,whch,thresh,opts] = get_args(varargin)
% Process inputs and do error-checking 
   
   % possible calling methods
   %  skl = seqkl(A)
   %  skl = seqkl(A,k)
   %  skl = seqkl(A,k,whch)
   %  skl = seqkl(A,k,opts)
   %  skl = seqkl(A,k,whch,opts)
   %  skl = seqkl(A,kmax,thresh)
   %  skl = seqkl(A,kmax,thresh,whch)
   %  skl = seqkl(A,kmax,thresh,opts)
   %  skl = seqkl(A,kmax,thresh,whch,opts)


   % arg 1 must be A
   if isa(varargin{1},'double')
      A = varargin{1};
   else
      error('A must be a real matrix.');
   end
   [m,n] = size(A);

   % arg 2 must be k (if it exists)
   if (nargin < 2)
      kmax = min(n,6);
   else
      kmax = varargin{2};
   end

   kstr = ['Requested basis size, k, must be a' ...
           ' positive integer <= n.'];
   if ~isa(kmax,'double') || ~isequal(size(kmax),[1,1]) || ~isreal(kmax) || (kmax>n) || (kmax<0),
      error(kstr)
   end
   if issparse(kmax)
      kmax = full(kmax);
   end
   if (ceil(kmax) ~= kmax)
      warning(msgid,['%s\n         ' ...
              'Non-integer k. Taking the ceiling.'],kstr)
      kmax = ceil(kmax);
   end

   % next argument may be either opts or whch or thresh: check if it's thresh
   if (nargin < 3)
      threshnotthere = 1;
      thresh = nan;
   else
      to = varargin{3};
      if isnumeric(to),
         threshnotthere = 0;
         thresh = upper(to);
      else
         threshnotthere = 1;
         thresh = nan;
      end
   end
   errstr = 'Threshhold must be a scalar >= 0.';
   if ~isequal(size(thresh),[1,1]),
      error(errstr);
   end

   % next argument may be either opts or thresh: check if it's thresh
   if (nargin >= 4-threshnotthere)
      wo = varargin{4-threshnotthere};
      if ischar(wo)
         whchnotthere = 0;
         whch = wo;
      else
         whchnotthere = 1;
         whch = 'L';
      end
   else
      whchnotthere = 1;
      whch = 'L'; 
   end
   errstr = 'whch must be ''L'' or ''S''.';
   if ~isequal(whch,'L') && ~isequal(whch,'S'),
      error(errstr);
   end

   if (nargin >= 5-threshnotthere-whchnotthere)
      opts = varargin{5-whchnotthere-threshnotthere};
      if ~isa(opts,'struct')
          error('Options argument must be a structure.')
      end
   else
      % create an empty struct to return
      opts = struct();
   end

   if (nargin > 5-threshnotthere-whchnotthere),
      % extra arguments sent in
      warning('Too many arguments.');
   end

end   % end function get_args
