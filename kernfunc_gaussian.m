function kh = kernfunc_gaussian(sigma)
%KERNFUNC_GAUSSIAN Returns handle to Gaussian kernel function.
%
%  KH = KERNFUNC_GAUSSIAN(SIGMA) returns a handle to a Gaussian kernel
%  function, also known as the squared exponential or radial basis
%  function. The input SIGMA is a scalar. The kernel is given by
%
%           KH = @(X,Y) exp( -((X-Y).^2) ./ (2.*SIGMA^2) )
% 
% The kernel can accept vector inputs, i.e., X & Y can be vectors.

kh = @(X,Y) exp(-((X-Y).^2)./(2.*sigma.^2));
