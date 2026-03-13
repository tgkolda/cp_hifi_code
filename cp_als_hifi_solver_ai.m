function W = cp_als_hifi_solver_ai(V,B,info)
%CP_ALS_HIFI_SOLVER_AI Solver for aligned infinite-dimensional mode in CP-HIFI.
%
%   W = CP_ALS_HIFI_SOLVER_AI(V,B,INFO) solves the system
%   (kron(V,K) + LAMBDA*I)*X = B for X, where K=INFO.KERNMAT and
%   LAMBDA=INFO.LAMBDA. Here, V is a symmetric matrix. 
%   The structure INFO has the following fields:
%      'solver'  - specifies the solver to use, and can any of
%                  ['direct', 'direct_decoupled', 'cg', 'pcg']
%      'kernmat' - the kernel matrix K
%      'lambda'  - the regularization parameter LAMBDA
%      'inmaxit' - (for 'cg' and 'pcg') maximum number of iterations
%      'intol'   - (for 'cg' and 'pcg') tolerance for the iterative solver
%      'U'       - (for 'pcg' and 'direct_decoupled') eigenvectors of K
%      'D'       - (for 'pcg' and 'direct_decoupled') eigenvalues of K
%  
% See also CP_ALS_HIFI.

% Code by Johannes Brust and Tamara Kolda, 2026

% Extract some info
solver = info.solver;
K = info.kernmat;
lambda = info.lambda;

if strcmp(solver,'direct')

    % Big matrix of Kronecker products, size rnk x rnk
    [nk,r] = size(B);
    rnk = r*nk;
    BigMat = kron(V,K) + lambda * eye(rnk);

    % Right hand side
    RHS = reshape(B,[],1);

    % Solve
    if info.nonneg
        Wvec = lsqnonneg(BigMat,RHS);
    else
        Wvec = BigMat \ RHS;
    end
    W = reshape(Wvec,nk,r);
    return;

end

% Algorithms
if strcmp(solver,'direct_decoupled')

    % This algorithm decouples the systme into block diagonal with
    % blocks d(i)*V + lambda*I. Each of the small systems
    % exploits that V is constant.
    U_K = info.U;
    d_K = diag(info.D);
    [U_V,D_V] = eig(V);
    d_V = diag(D_V);
    DINV = 1 ./ (d_K*d_V' + lambda);
    W = U_K*((U_K'*B*U_V).*DINV)*U_V';
    return;
    
end

if ismember(solver,{'cg','pcg'})

    %opts.trace   = 0;
    %opts.printit = false;    
    maxit       = info.inmaxit;
    tol         = info.intol;
    if isempty(maxit)  
        maxit   = 50;           
    end
    if isempty(tol)    
        tol     = 1e-4;         
    end

    switch solver
        case 'cg'
            [n,r] = size(B);
            RHS = reshape(B,[],1);
            matvec = @(x)(reshape(K*reshape(x,n,r)*V,[],1)+lambda*x);
            [Wvec,~] = pcg(matvec,RHS,tol,maxit);            
        
        case 'pcg'
            
            % Initializations
            U_K = info.U;
            d_K = diag(info.D);        
            [n,r] = size(B);
            
            % Preconditioning
            Delta = reshape(d_K*diag(V)' + lambda,[n*r,1]);
            prcnd = @(x)( x ./ Delta );
            
            % Right hand side and solve
            matvec      = @(x)(reshape(d_K.*reshape(x,n,r)*V,[],1)+lambda*x);                
            RHS         = reshape(U_K'*B,[],1);            
            [Wvec1,~]   = pcg(matvec,RHS,tol,maxit,prcnd);            
            
            % transform the solution back to initial space    
            Wvec = U_K*reshape(Wvec1,[n,r]);
                
    end
    r   = size(V,1);
    nk  = size(B,1);
    W   = reshape(Wvec,nk,r);
    return;
end

error('Unknown solver (%s) specified.', solver);

end
