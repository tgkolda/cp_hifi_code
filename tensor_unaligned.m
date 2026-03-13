classdef tensor_unaligned < sptensor
    %TENSOR_UNALIGNED Class for subsampled tensor with unaligned modes.
    %
    %   This class is dervied from sptensor. The sampled entries are stored
    %   as an incomplete tensor, based on the sptensor class. The
    %   tensor_unaligned adds a mapping from the indices of this tensor to
    %   the original tensor indices (or corresponding x-values).
    %
    %   See also TENSOR_ALIGNED, KTENSOR_HIFI, CP_ALS_HIFI.
    
    % Code by Tamara Kolda, 2025. 
    
    properties
        xvals % x-values for each mode, usually 1:size(X,1)
    end

    methods

        function obj = tensor_unaligned(varargin)
            %TENSOR_UNALIGNED Constructor for tensor_unaligned class.
            %
            %   TU = TENSOR_UNALIGNED(X,N) creates an unaligned tensor
            %   based on sampling N entries from the tensor X.
            %
            %   TU = TENSOR_UNALIGNED(X,N,XVALS) stores the x-values that
            %   correspond to each mode. This is ndims(X) x 1 cell array
            %   such that XVALS{k} holds and 1 x size(X,k) array of
            %   strictly increasing values.
            %
            %   TU = TENSOR_UNALIGNED(SUBS,VALS,SIZE,XVALS) gives the
            %   components of the unaligned tensor. SUBS is a Q x D array
            %   where Q is the number of entries and D is the number of
            %   dimensions. VALS is a Q x 1 array containing the
            %   corresponding values for the entries specified by SUBS. 
            %   SIZE is a D x 1 specification of the tensor size. XVALS is
            %   D x 1 cell array where the cell K has an array of length
            %   SZ(K) specifying the x-values corresponding to each index
            %   in mode K.
 

            if nargin == 0

                newsubs = [];
                newvals = [];
                newsize = [];
                newxvals = [];
            
            elseif nargin == 4 % subs, vals, size, xvals
                
                newsubs = varargin{1};
                newvals = varargin{2};
                newsize = varargin{3};
                newxvals = varargin{4};

            else % X,nsamp,xvals

                X = varargin{1};
                nsamp = varargin{2};
                if ~isa(X,'tensor')
                    error('X must be a tensor');
                end

                d = ndims(X);
                N = prod(X.size);

                if N > 2^(53) - 1
                    error('Total size of tensor is too large for sampling');
                end
                if nsamp > N
                    error('Number of samples is larger than size of tensor');
                end

                % Set the x-values for the tensor X
                if nargin < 3
                    if isa(X,'tensor_aligned')
                        xvals = X.xvals;
                    else
                        xvals = cell(1,d);
                        for k = 1:d
                            xvals{k} = 1:size(X,k);
                        end
                    end
                else
                    for k = 1:d
                        if length(xvals{k}) ~= size(X,k)
                            error('length(xvals{%d}) is not the same as size(X,%d)',k,k)
                        end
                        if any(diff(xvals{k}) <= 0)
                            error('xvals{%d} is not strictly increasing',k)
                        end
                    end
                end

                % Sample linear indices
                sampled_idx = (randperm(N,nsamp))';

                % Extract values
                newvals = X(sampled_idx);

                % Convert linear indices to tuples
                sampled_subs = tt_ind2sub(size(X),sampled_idx);

                % Remove indices that are not in the sample and re-index,
                % including updating the x-vals.
                newsubs = zeros(size(sampled_subs));
                newxvals = cell(d,1);
                newsize = zeros(1,d);
                for k = 1:d
                    [idx,~,newsubs(:,k)] = unique(sampled_subs(:,k));
                    newxvals{k} = xvals{k}(idx);
                    newsize(k) = length(newxvals{k});
                end
            end
            obj = obj@sptensor(newsubs,newvals,newsize,[],'incomplete');
            obj.xvals = newxvals;

        end

        function obj = set_mode_xvals(obj,k,vals)
            % SET_MODE_XVALS Set the x-values for a mode of the tensor.

            if length(nvals) ~= size(obj,k)
                error('Length of vals must equal size(obj,k)');
            end
            if ~isvector(vals) || any(diff(vals) <= 0)
                error('vals must be a row vector of strictly increasing values');
            end
            if ~isrow(vals)
                vals = vals';
            end

            obj.vals{k} = vals;

        end % Function set_mode_xvals

        function ta = full(obj)
            %FULL Convert tensor_unaligned to tensor_aligned object.
            %
            %   TA = FULL(TU) converts the tensor_unaligned TU to a
            %   TA, a tensor_aligned object (aka a dense tensor with
            %   corresponding x-values) with nan for the missing values.
            X = full@sptensor(obj);
            ta = tensor_aligned(X);
            for k = 1:ndims(obj)
                ta = set_mode_xvals(ta,k,obj.xvals{k});
            end
        end

        function s = saveobj(obj)
            %SAVEOBJ Save tensor_unaligned object.
            s.subs = obj.subs;
            s.vals = obj.vals;
            s.size = obj.size;
            s.xvals = obj.xvals;
        end

    end

    methods (Static)
        function obj = loadobj(s)
            %LOADOBJ Load tensor_unaligned object.
            obj = tensor_unaligned();
            obj.subs = s.subs;
            obj.vals = s.vals;
            obj.size = s.size;
            obj.xvals = s.xvals;
        end

    end
end