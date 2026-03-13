function A = cp_als_hifi_solver_af(V,B,info)
%CP_ALS_HIFI_SOLVER_AF Aligned finite-dimensional mode solver for CP-HIFI.
%
%   A = CP_ALS_HIFI_SOLVER_AF(V,B,INFO) solves the linear least squares 
%   problem min || B - A*V ||_F^2 for A, where V and B are given matrices
%   and INFO is a struct with the field 'nonneg' that indicates whether to 
%   enforce nonnegativity constraints on the solution A.
%
%   See also CP_ALS_HIFI.

% Code by Johannes Brust and Tamara Kolda, 2026


nonneg = info.nonneg;

% Compute the matrix of coefficients for linear system
if nonneg
    [nk,r] = size(B);
    A = zeros(nk,r);
    for i = 1:nk
        tmp = lsqnonneg(V,B(i,:)');
        A(i,:) = tmp;
    end
else
    A = B / V;
end