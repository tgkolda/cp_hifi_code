function W = cp_als_hifi_solver_ui(X,A,kmode,info)
%CP_ALS_HIFI_SOLVER_UI Solver for unaligned infinite mode for CP-HIFI.
%
%   W = CP_ALS_HIFI_SOLVER_UI(X,A,KMODE,INFO) solves the least squares 
%   problem in mode KMODE for scarce tensor X and factor matrices in the 
%   cell array A. INFO is a structure that contains key information:
%      'solver'  - string specifying which solver to use
%      'kernmat' - kernel matrix for infinite mode
%      'lambda'  - regularization parameter
%      'rho'     - 2nd regularization parameter (not needed for direct)
%      'nonneg'  - boolean flag for nonnegativity constraints 
%      'U', 'D'  - eigenvectors/values of kernel matrix (for pcg)
%      'maxit'   - maximum number of iterations (for iterative solvers)
%      'tol'     - tolerance for stopping criterion (for iterative solvers)
%
%   See also CP_ALS_HIFI.

% Code by Johannes Brust and Tamara Kolda, 2026


solver = info.solver;
if ismember(solver,{'direct','direct_nonsym'})
    W = sub_ui_direct_nonsym(X,A,kmode,info);
    return;
end

if ismember(solver,{'direct_sym'})
    W = sub_ui_direct_sym(X,A,kmode,info);
    return;
end

% algorithms for normal equations
if ismember(solver,{'pcg','cg','pcg_v1','pcg_v2','pcg_v3'})    
    W = sub_ui_sq(X,A,kmode,info);
end

end

function W = sub_ui_sq(T,A,kmode,info)
%SUB_UI_SQ Solves unaligned infinite-dimensional system in CP-HIFI.
%
%   W = SUB_UI_SQ(T,A,KMODE,INFO) uses an iterative method to solve the 
%   CP-HIFI subproblem in mode KMODE for scarce tensor T and factor 
%   matrices in the cell array A. INFO is a structure with fields:
%      'kernmat' - kernel matrix for infinite mode
%      'U', 'D'  - eigenvectors/values of kernel matrix
%      'lambda'  - regularization parameter
%      'rho'     - 2nd regularization parameter
%      'solver'  - string specifying which iterative solver to use
%      'inmaxit' - maximum number of iterations
%      'intol'   - tolerance for stopping criterion
%
%   See also CP_ALS_HIFI, MODE_SOLVER_UI.

% Extract from Inputs

% Get information about X
q = nnz(T); % Number of known entries
d_K = ndims(T);
n = size(T,kmode);
idx_kmode = T.subs(:,kmode);

% Get rank of factorization
if kmode > 1
    r = size(A{1},2);
else 
    r = size(A{2},2);
end

% Extract RKHS and solver info
K = info.kernmat;
U_K = info.U;
D_K = info.D;
lambda  = info.lambda;
rho = info.rho;
solver = info.solver;
maxit = info.inmaxit;
tol = info.intol;

% Fill for any missing info
if isempty(rho)
    rho    = 0;            
end
if isempty(solver) 
    solver = 'pcg';        
end
if isempty(maxit)  
    maxit  = 50;           
end
if isempty(tol)    
    tol    = 1e-4;         
end

% Form Zexp, the expanded version of the Khatri-Rao product
Zexp = ones(q,r);
for k = [1:kmode-1,kmode+1:d_K]         
    Zexp = Zexp .* A{k}(T.subs(:,k),:);
end

% MTTKRP B = Tk*Z where Tk is the unfolded tensor T in mode KMODE
B = zeros(n,r);
for j = 1:r
    vj = accumarray(idx_kmode, T.vals .* Zexp(:,j), [n 1]);
    B(:,j) = vj;
end

% Solve
if strcmp(solver,'cg')   

    RHS = reshape(K*B,[],1);
    matvec = @(x)( sub_ui_sq_matvec(x,Zexp,idx_kmode,K,lambda,rho) );
    [Wvec,~] = pcg(matvec,RHS,tol,maxit);

elseif strcmp(solver,'pcg')

    % preconditioner using a multiple of the identity 
    % approximation for the projection S*S' approx dens*I

    RHS = reshape(K*B,[],1);
    V = ones(r,r);
    for k = [1:kmode-1,kmode+1:d_K]         
        V = V .* (A{k}'*A{k});
    end
    [U_V,D_V] = eig(V);
    density = q/prod(size(T));
    d_V = diag(D_V);
    d_K = diag(D_K);
    DINV = 1./((density*(d_K.*d_K)*d_V')+(lambda*d_K*ones(1,r))+rho);
    prcnd = @(x) reshape(U_K*((U_K'*reshape(x,n,r)*U_V).*DINV)*U_V',[],1);
    matvec = @(x) sub_ui_sq_matvec(x,Zexp,idx_kmode,K,lambda,rho);
    [Wvec,~] = pcg(matvec,RHS,tol,maxit,prcnd);

elseif strcmp(solver,'pcg_v1')
    
    % working in transformed space, with kernel in factored form
    % full diagonal preconditioner
    % Hexp = H(idx_k,:); 
    % matdiagFtF = (Hexp.^2)'*(Zexp.^2);    
    % Delta = matdiagFtF(:) + lambda * repmat(diagD,[r 1]) + rho;

    RHS     = reshape(D_K*U_K'*B,[],1);
    diagD   = diag(D_K);        
    H       = U_K*D_K;
    for j = 1:r
        vj = accumarray(idx_kmode, Zexp(:,j) .* Zexp(:,j), [n 1]);
        VH = sqrt(vj).* H;
        B(:,j) = sum(VH.*VH,1);
    end
    Delta = B(:) + lambda * repmat(diagD,[r 1]) + rho;    
    prcnd = @(x) ( x ./ Delta );
    matvec = @(x) sub_ui_sq_matvec2(x,Zexp,B,H,diagD,lambda,rho,idx_kmode);    
    [Wvec1,~] = pcg(matvec,RHS,tol,maxit,prcnd);
    Wvec = U_K*reshape(Wvec1,[n,r]);

elseif strcmp(solver,'pcg_v2')

    % GPT 5.2 pro
    % preconditioner without the projection matrix

    RHS = reshape(K*B,[],1);
    V = ones(r,r);
    for k = [1:kmode-1,kmode+1:d_K]         
        V = V .* (A{k}'*A{k});
    end
    In = eye(n);
    Ir = eye(r);
    prcnd = @(x) reshape(((K+rho*In)\reshape(x,n,r))/(V+lambda*Ir),[],1);
    matvec = @(x) sub_ui_sq_matvec(x,Zexp,idx_kmode,K,lambda,rho);
    [Wvec,~] = pcg(matvec,RHS,tol,maxit,prcnd);

elseif strcmp(solver,'pcg_v3')

    % Diagonal preconditioner w/o transforming the system    

    RHS = reshape(K*B,[],1);
    Kexp = K(idx_kmode,:); 
    matdiagFtF = (Kexp.^2)'*(Zexp.^2);    
    Delta = matdiagFtF(:) + lambda * repmat(diag(K),[r 1]) + rho;
    prcnd = @(x)( x ./ Delta );
    matvec = @(x)( sub_ui_sq_matvec(x,Zexp,idx_kmode,K,lambda,rho) );
    [Wvec,~] = pcg(matvec,RHS,tol,maxit,prcnd);

end

W = reshape(Wvec,n,r);

end

function y = sub_ui_sq_matvec(x,Zexp,idx,K,lambda,rho)
%SUB_UI_SQ_MATVEC Matvec product for CP-HIFI unaligned infinite case.
%
%   Y = SUB_UI_SQ_MATVEC(X,ZEXP,IDX,K,LAMBDA,RHO) computes the product
%   Y = ((kron(Z,K)'*S*S'*kron(Z,K) + LAMBDA*kron(I,K) + RHO*I) * X.
%   The matrix S is a selection matrix that selects Q rows of the 
%   kron(Z,K) matrix where Z is an M x R Khatri-Rao product of a series of
%   factor matrices and K is an N x N kernel matrix. The matrix S is not 
%   explicitly specified given but instead represented implicitly by the
%   combination of the precomputed Zexp of size Q x R and the Q-array of 
%   indices in [N] stored as IDX. The input X is an N x R matrix stored as 
%   a vector. The output Y is also an N x R matrix stored as a vector. 
%   LAMBDA > 0 and RHO > 0 are regularization parameters.
%
%   See also SUB_UI_SQ.

r = size(Zexp,2);
n = size(K,1);
X = reshape(x,n,r);

Y = K*X;

% compute the product S'*kron(Zexp,I)*(KX). Y(idx,:) generates a
% tall matrix but does not incur any flops.
V = sum(Zexp.*Y(idx,:), 2);

% compute (kron(Zexp,I)'*S)*V, which only incurs additions within
% the accumarray itself.
B = zeros(n,r);
for j = 1:r
    B(:,j) = accumarray(idx, V.*Zexp(:,j), [n 1]);
end

Y = K'*B + lambda*Y + rho.*X;
y = reshape(Y,[],1);

end

function y = sub_ui_sq_matvec2(x,A,B,K,D,lambda,rho,idx)
%SUB_UI_SQ_MATVEC2 Matrix vector product for CP-HIFI unaligned infinite case.
%
%   Y = SUB_UI_SQ_MATVEC2(X,A,B,K,D,LAMBDA,RHO,IDX) computes the product
%   Y = (RHO*I + LAMBDA*kron(I,K) + ( (S'*kron(A,K))'*S'*kron(A,K))*X
%   Here, X is vec(X) where X is a matrix of size (size(B,1) x size(A,2)).
%   The selection matrix S is not explicit, but implicitly represented by
%   the indices in IDX. B is a memory buffer in order to avoid memory
%   reallocation for every matrix-vector product. A has the same number of rows
%   as IDX and K is a square psd kernel matrix or a square matrix representing
%   U'*K. LAMBDA > 0 and RHO > 0 are regularization parameters and the D
%   is either empty or a vector representing diagonal elements

%TODO: Change name to SUB_UI_MATVEC 


r   = size(A,2);
n   = size(B,1);
X   = reshape(x,n,r);

% matrix K may be the symmetric kernel or the product of two factors 
% of an eigendecomposition, i.e., U*D
y   = (K * X);

if ~isempty(D)
    y1  = lambda * (D .* X);
else
    y1  = lambda * y;
end

% compute the product S'*kron(A,I)*(KX). Y(idx,:) generates are
% tall matrix but does not incur any flops.
z = sum(A.*y(idx,:),2);

% compute (kron(A,I)'*S)*z, which only incurs additions within
% the accumarray itself.
for j = 1:r
    B(:,j) = accumarray(idx, z .* A(:,j), [n 1]);     
end

y = y1 + rho.*X + K'*B;
y = reshape(y,[],1);
    
end

function W = sub_ui_direct_sym(X,A,kmode,info)
%HIFI_MODE_SOLVER Solves for weights of HIFI mode.

KerMat = info.kernmat;
lambda = info.lambda;
rho = info.rho;
nonneg = info.nonneg;

% X = tensor
% A = cell array of factor matrices
% kmode = mode being solved
% KerMat = Kernel matrix
% lambda = regularization

% Get all the other relevant sizes
d = ndims(X);
nz = nnz(X);
skiprng = [1:kmode-1,kmode+1:d];
p = size(KerMat,1);
if kmode > 1
    r = size(A{1},2);
else
    r = size(A{2},2);
end

% Big matrix of Kronecker products, size np x np
Zexp = ones(nz,r);
for k = skiprng
    Akexp = A{k}(X.subs(:,k),:);
    Zexp = Zexp .* Akexp;
end

% LFAC_TR = khatrirao(Zexp.',Ink(:,X.subs(:,kmode)));
F = khatrirao(Zexp.',KerMat(:,X.subs(:,kmode)));
BigMat = rho * eye(r*p) + lambda * kron(eye(r),KerMat) +  F*F';

RHS = zeros(p,r);
for j = 1:r
    RHS(:,j) = accumarray(X.subs(:,kmode), X.vals .* Zexp(:,j), [p 1]);
end
RHS = reshape(KerMat*RHS,[],1);

% Solve
if nonneg
    Wvec = lsqnonneg(BigMat,RHS);
else
    Wvec = BigMat \ RHS;
end
W = reshape(Wvec,p,r);

end

function W = sub_ui_direct_nonsym(T,A,kmode,info)
%SUB_UI_DIRECT_NONSYM Solver for unaligned infinite mode for CP-HIFI.
%
%   W = SUB_UI_DIRECT_NONSYM(T,A,KMODE,INFO) uses a direct method to solve
%   an unsymmetric version of the CP-HIFI subproblem  in mode KMODE for 
%   scarce tensor T and factor matrices in the cell array A. INFO is a 
%   structure with fields:
%      'kernmat' - kernel matrix for infinite mode
%      'lambda'  - regularization parameter
%      'nonneg'  - boolean flag for nonnegativity constraints
%
%   See also CP_ALS_HIFI, MODE_SOLVER_UI.

KerMat = info.kernmat;
lambda = info.lambda;
nonneg = info.nonneg;

% Get the relevant sizes
d = ndims(T);
q = nnz(T);
n = size(KerMat,1);
if kmode > 1
    r = size(A{1},2);
else
    r = size(A{2},2);
end

% Assemble BigMat and RHS
In = eye(n);

% Big matrix of Kronecker products, size np x np
Zexp = ones(q,r);
for k = [1:kmode-1,kmode+1:d]
    Zexp = Zexp .* A{k}(T.subs(:,k),:);
end

Gt = khatrirao(Zexp.',In(:,T.subs(:,kmode)));
Ft = khatrirao(Zexp.',KerMat(:,T.subs(:,kmode)));
BigMat =  Gt*Ft' + lambda * eye(r*n);
RHS = Gt * T.vals;

% Solve
if nonneg
    Wvec = lsqnonneg(BigMat,RHS);
else
    Wvec = BigMat \ RHS;
end
W = reshape(Wvec,n,r);

end
