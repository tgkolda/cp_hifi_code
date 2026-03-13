function B = cp_als_hifi_solver_uf(X,A,kmode,info)
%CP_ALS_HIFI_SOLVER_UF Solve for unaligned finite mode for CP-HIFI.
%
%   B = CP_ALS_HIFI_SOLVER_UF(X,A,KMODE) solves the least squares problem 
%   in mode KMODE for scarce tensor X and factor matrices in the cell 
%   array A. The problem is decomposed into a series of smaller problems. 
%
%   B = CP_ALS_HIFI_SOLVER_UF(X,A,KMODE,INFO) solves the same problem, but 
%   with additional options specified in the INFO structure. If INFO.NONNEG
%   is true, nonnegativity constraints are enforced on the solution. 
%
%   See also CP_ALS_HIFI.

% Code by Johannes Brust and Tamara Kolda, 2026

% The bottom comments include the parfor version of the method

% If INFO.SOLVER is 'direct', then the old "full" direct solver is used.
% This is not advertised nor recommended and retained only for testing 
% purposes. Subroutine below.
if (nargin == 4) && isfield(info,'solver') && strcmp(info.solver,'direct')
    B = sub_uf_direct(X,A,kmode,info);
    return;
end

if nargin < 4
    nonneg = false;
else
    nonneg = info.nonneg;
end


% Extract scarce data
Xsubs = X.subs;
Xvals = X.vals;

% Get all the relevant sizes
nz = size(Xsubs,1);
d = size(Xsubs,2);
p = size(X,kmode);
skiprng = [1:kmode-1,kmode+1:d];
if kmode > 1
    r = size(A{1},2);
else
    r = size(A{2},2);
end

% Direct algorithm (sequential)
% Matrices for Khatri-Rhao product and right hand side
Z   = ones(nz,r);
for k = skiprng
    Akexp = A{k}(Xsubs(:,k),:);
    Z     = Z .* Akexp;
end

% Decompose the problem into p small least squares problems
B = zeros(p,r);
for i = 1:p        
    IDX = find(Xsubs(:,kmode)==i);
    if nonneg
        B(i,:) = lsqnonneg(Z(IDX,:), Xvals(IDX));
    else
        if length(IDX) > r
            B(i,:) = Z(IDX,:) \ Xvals(IDX);
        else
            B(i,:) = lsqminnorm(Z(IDX,:), Xvals(IDX));
        end
    end
end

% % Direct algorithm (parallel)
% % Matrices for Khatri-Rhao product and right hand side
% t2 = tic;
% Z = ones(nz,r);
% for k = skiprng
%     Akexp = A{k}(Xsubs(:,k),:);
%     Z = Z .* Akexp;
% end
% KK  = Xsubs(:,kmode);
% 
% B1  = zeros(p,r);
% 
% ZZ  = cell(p,1);
% BB  = cell(p,1);
% ID  = cell(p,1);
% for i=1:p
%     ID{i} = find(KK==i);
%     ZZ{i} = Z(ID{i},:);
%     BB{i} = Xvals(ID{i});
% end
% 
% % parallel loop
% parfor i=1:p
% 
%     B1(i,:) = ZZ{i}\ BB{i};
% 
% end
% t2 = toc(t2);

end 

function B = sub_uf_direct(X,A,kmode,info)
%SUB_UF_DIRECT Direct solver for unaligned CP-HIFI subproblem.
%
%   This is the "old" method that forms and solves the full normal equations
%   for the unaligned finite mode.

% INPUTS:
% X = scarce tensor
% A = cell array of factor matrices
% kmode = mode being solved
% info = has field 'nonneg' (true/false)

nonneg = info.nonneg;

% Extract scarce data
Xsubs = X.subs;
Xvals = X.vals;
sz = size(X);

% Chunk the nonzeros (will need to make this an optional paramater)
nz = size(Xsubs,1);
nzchunksize = 5000;
cstart= 1:nzchunksize:nz;
cend = [cstart(2:end)-1 nz];
nchunks = length(cstart);
csz = cend-cstart+1;

% Get all the other relevant sizes
d = size(Xsubs,2);
skiprng = [1:kmode-1,kmode+1:d];
p = sz(kmode);
if kmode > 1
    r = size(A{1},2);
else
    r = size(A{2},2);
end

%
Ink = eye(p);

% Big matrix of Kronecker products, size np x np
BigMat = zeros(r*p);
RHS = zeros(r*p,1);

for c = 1:nchunks
    chunk = cstart(c):cend(c);
    Hchunk = ones(csz(c),r);
    for k = skiprng
        Akexp = A{k}(Xsubs(chunk,k),:);
        Hchunk = Hchunk .* Akexp;
    end
    KK = Xsubs(chunk,kmode);
    InkChunk_Tr = Ink(:,KK);
    FAC_TR = khatrirao(Hchunk',InkChunk_Tr);
    BigMat = BigMat + FAC_TR*FAC_TR';
    RHS = RHS + FAC_TR*Xvals(chunk);
end

% Solve normal equations of size np x np
if nonneg
    Bvec = lsqnonneg(BigMat,RHS);
else
    Bvec = BigMat \ RHS;
end
B = reshape(Bvec,p,r);

end