function [P,output] = cp_als_hifi(X,R,hifiinfo,varargin)
%CP_ALS_HIFI CP-ALS with HIFI.
%
%   K = CP_ALS_HIFI(X,R,HIFIINFO) computes a rank-R CP decomposition of 
%   the aligned or unaligned tensor X. HIFIINFO is length-d cell array, 
%   each element a structure with the following fields:
%     'inf'    - TRUE if infinite-dimensional mode, FALSE otherwise 
%     'kfunc'  - handle to kernel function taking 2 vector inputs
%     'lambda' - regularization parameter (optional, default 0.1)
%     'rho'    - second regularization parameter (optional, default 0.1)
%   The result is returned as a KTENSOR_HIFI K.
%
%   K = CP_ALS_HIFI(X,R,HIFIINFO,'param',value,...) specifies optional
%   parameters and their values. Parameters include:
%      'tol'      - Tolerance on difference in fit {1.0e-4}
%      'abserr'   - Absolute error tolerance {1.0e-6}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%      'init'     - Initial guess [{'random'}|'nvecs'|cell array]
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%      'fixsigns' - Fix signs of the columns of the factor matrices {true}
%      'nonneg'   - Constrain to be nonnegative {false}
%      'solver'   - Inner solver for infinite-dimensional modes, 
%                   ['pcg'|{'direct_decoupled'}|'direct'] for aligned or 
%                   [{'pcg'}|'direct_sym'|'direct_nonsym'] for unaligned
%      'inmaxit'  - Maximum number of iterations for pcg inner solver {50}
%      'intol'    - Relative residual for inner solver {1e-4}     
%      'trace'    - Record fit and time at each iteration {false}
%
%   [K,OUTPUT] = CP_ALS_HIFI(X,R,HIFIINFO,...) also returns an OUTPUT
%   structure with the following fields:
%      'params'     - Parameters used (structure)
%      'iters'      - Number of iterations taken
%      'final_fit'  - Final fit
%      'eigtime'    - Time to compute eigendecompositions (vector)
%      'init_time'  - Time to initialize (if 'trace'==true)
%      'time_trace' - Running time at each iteration (if 'trace'==true)
%      'fit_trace'  - Fit at each iteration (if 'trace'==true)
%      
%   See also CP_ALS, KTENSOR_HIFI, TENSOR_ALIGNED, TENSOR_UNALIGNED.
%
%Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

% Code by Johannes Brust and Tamara Kolda, 2026

% Undocumented option for UI solvers: ['pcg_v1','pcg_v2','pcg_v3'] for 
% other preconditioners.

%TODO: We should eventually allow for different nonneg per mode.
%TODO: Need to support nonnegativity for continuous modes for all solvers.
%TODO: Rename 'direct' to 'direct_decoupled' and "direct_old" to "direct_full" (or whatever matches the final paper naming)

%% Some defaults
lambda_default = 0.1;
rho_default = 0.1;

%% Extract number of dimensions and norm of X.
d = ndims(X);
normX = norm(X);

aligned = isa(X, 'tensor_aligned');
unaligned = isa(X, 'tensor_unaligned');
if ~(aligned || unaligned)
    error('X must be a tensor_aligned or tensor_unaligned object');
end

%% Set algorithm parameters from input or by using defaults
if aligned
    default_solver='direct_decoupled'; 
else
    default_solver='pcg'; 
end
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('abserr',1e-6,@isscalar);
params.addParameter('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:d,@(x) isequal(sort(x),1:d));
params.addParameter('init','random', @(x) (iscell(x) || ismember(x,{'random','rand','randn'})));
params.addParameter('printitn',1,@isscalar);
params.addParameter('fixsigns',true,@islogical);
params.addParameter('trace',false,@islogical);
params.addParameter('nonneg',false,@islogical);
params.addParameter('solver',default_solver);
params.addParameter('inmaxit',50);
params.addParameter('intol',1e-4);

params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
abserr = params.Results.abserr;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
dotrace = params.Results.trace;
nonneg = params.Results.nonneg;
solver = params.Results.solver;
inmaxit = params.Results.inmaxit;
intol = params.Results.intol;

if isscalar(inmaxit)
    inmaxit = repmat(inmaxit,d,1);
elseif numel(inmaxit) ~= d
    error('OPTS.inmaxit must be a scalar or a vector of length %d',d);
end

if isstring(solver) || ischar(solver)
    solver = repmat({solver},d,1);
elseif iscell(solver)
    if numel(solver) ~= d
        error('OPTS.solver must be a string or a cell array of length %d',d);
    end
else
    error('OPTS.solver must be a string or a cell array of length %d',d);
end

if isscalar(intol)
    intol = repmat(intol,d,1);
elseif numel(intol) ~= d
    error('OPTS.intol must be a scalar or a vector of length %d',d);
end

%% Error checking 

if ~iscell(hifiinfo) || (numel(hifiinfo) ~= d)
    error('HIFIINFO must be a cell array of length %d',d);
end

%% Set up for HIFI
xvals = X.xvals;
eigtime = zeros(d,1);
for k = 1:d

    hifiinfo{k}.nonneg = nonneg;
    
    if hifiinfo{k}.inf == true
        
        hifiinfo{k}.solver = solver{k};
        hifiinfo{k}.kernmat = hifiinfo{k}.kfunc(xvals{k}',xvals{k});          
        hifiinfo{k}.inmaxit = inmaxit(k);
        hifiinfo{k}.intol   = intol(k);
    
        if ~isfield(hifiinfo{k},'lambda')
            hifiinfo{k}.lambda = lambda_default;
        end
        if ~isfield(hifiinfo{k},'rho')
            hifiinfo{k}.rho = rho_default;
        end

        if ismember(solver{k},{'pcg','pcg_v1'}) || ...
            (aligned && ismember(solver{k},{'direct_decoupled'}))      
            et = tic;     
            [U,D] = eig(hifiinfo{k}.kernmat); 
            eigtime(k) = toc(et);
        else           
            U = [];
            D = [];
        end
        hifiinfo{k}.U = U;
        hifiinfo{k}.D = D;

        if nonneg && ~ismember(solver,{'direct'})
            warning(['Nonnegativity for continuous modes is only supported by ' ...
                'direct_old solvers (mode',num2str(k),')']);
        end
    else
        if ismember(solver{k}, {'direct_sym', 'direct_nonsym','direct'})
            hifiinfo{k}.solver = 'direct';
        end
    end

end

%% Set up and error checking on initial guess for A/W (AWinit)
if iscell(init)
    AWinit = init;
    if numel(AWinit) ~= d
        error('OPTS.init does not have %d cells',d);
    end
    for k = dimorder(2:end)
        if ~isequal(size(AWinit{k}),[size(X,k) R])
            error('OPTS.init{%d} is the wrong size',k);
        end
    end
else
    AWinit = cell(d,1);
    if strcmp(init,'random') || strcmp(init,'rand')
        for k = dimorder(2:end)
            AWinit{k} = rand(size(X,k),R);
        end
    elseif strcmp(init,'randn')
        for k = dimorder(2:end)
            AWinit{k} = randn(size(X,k),R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing A, AtA, weightmat, and the fit.
initstart = tic;
A = cell(d,1);
weightmat = cell(d,1);

for k = 1:d
    if hifiinfo{k}.inf 
        weightmat{k} = AWinit{k};
        if k ~= dimorder(1)
            A{k} = hifiinfo{k}.kernmat * weightmat{k};
        end
    else 
        A{k} = AWinit{k};
   end
end

fit = 0;

if aligned 
    AtA = zeros(R,R,d);
    for k = 1:d
        if ~isempty(A{k})
            AtA(:,:,k) = A{k}'*A{k};
        end
    end
end

if printitn>0
    fprintf('CP_ALS_HIFI (Hybrid Finite/Infinite Dimensional):\n');
    fprintf(' %s of size %s', class(X), mat2str(size(X)));
    if isa(X,'sptensor')
        fprintf(', sparse with %d (%.2g%%) nonzeros', ...
            nnz(X), 100*nnz(X)/prod(size(X))); %#ok<PSIZE>
    end
    fprintf('\n');
    fprintf(' infinite modes = %s\n', mat2str(cellfun(@(x) x.inf, hifiinfo)));
    fprintf(' R = %d, maxiters = %d, tol = %e\n', R, maxiters, fitchangetol);
    if ~isequal(dimorder,1:d)
        fprintf(' dimorder = %s, ', mat2str(dimorder));
    end
    fprintf(' init = %s, ', init);
    fprintf(' nonneg = %d\n', nonneg);
    fprintf(' solver (inner) = %s\n',strjoin(solver, ', '));
    if ~ismember(solver,{'direct_decoupled','direct'})
        fprintf(' maxit  (inner) = %s\n',strjoin(compose('%d',inmaxit), ', '));
        fprintf('   tol  (inner) = %s\n',strjoin(compose('%.3g',intol), ', '));
    end
end

% Set up kernel functions cell array
kfunc = cell(d,1);
for k = 1:d
    if hifiinfo{k}.inf
        kfunc{k} = hifiinfo{k}.kfunc;
    end
end


if dotrace
    inittime = toc(initstart);
    fittrace = zeros(maxiters,1);
    timetrace = zeros(maxiters,1);
    itertime = tic;
end


%% Main Loop: Iterate until convergence


for iter = 1:maxiters

    fitold = fit;

    % Iterate over all N modes of the tensor
    for k = dimorder(1:end)
        
        if aligned 
        
            V = prod(AtA(:,:,[1:k-1 k+1:d]),3);           
            B = mttkrp(X,A,k);

            if hifiinfo{k}.inf 
                W = cp_als_hifi_solver_ai(V,B,hifiinfo{k});
                Anew = hifiinfo{k}.kernmat * W;
            else  
                W = [];
                Anew = cp_als_hifi_solver_af(V,B,hifiinfo{k});                
            end
        
        else 

            if hifiinfo{k}.inf 
                W = cp_als_hifi_solver_ui(X,A,k,hifiinfo{k});
                Anew = hifiinfo{k}.kernmat * W;
            else 
                W = []; 
                Anew = cp_als_hifi_solver_uf(X,A,k,hifiinfo{k});
            end

        end

        % Rescaling
        if iter == 1
            lambda = sqrt(sum(Anew.^2,1))'; %2-norm
        else
            lambda = max( max(abs(Anew),[],1), 1 )'; %max-norm
        end
        lambda = max(lambda,1e-10); % Don't let all-zero rows break things
        Anew = bsxfun(@rdivide,Anew,lambda');
        if hifiinfo{k}.inf 
            W = bsxfun(@rdivide,W,lambda');
        end
        
        % Save results
        weightmat{k} = W;
        A{k} = Anew;       

        % Update AtA
        if aligned
            AtA(:,:,k) = A{k}'*A{k};
        end
    
    end

    % Assemble final answer
    P = ktensor_hifi(lambda,A,xvals,kfunc,weightmat);
    if aligned
        normresidual = norm(X-full(P));
    else
        normresidual = norm_masked_diff(P,X);
    end
    
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);

    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol) || (fit > 1 - abserr)
        flag = 0;
    else
        flag = 1;
    end

    if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
    end

    if dotrace
        fittrace(iter) = fit;
        timetrace(iter) = toc(itertime);
    end

    % Check for convergence
    if (flag == 0)
        break;
    end
end

%% Clean up the final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
if params.Results.fixsigns
    P = fixsigns(P);
end
%% 
if printitn>0
    fprintf(' Final f = %e \n', fit);
end

output = struct;
output.params = params.Results;
output.iters = iter;
output.final_fit = fit;
output.eigtime = eigtime;
if dotrace
    output.init_time = inittime;
    output.time_trace = timetrace(1:iter);
    output.fit_trace = fittrace(1:iter);
end

end

