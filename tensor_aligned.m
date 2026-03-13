classdef tensor_aligned < tensor
    %TENSOR_ALIGNED Class for subsampled tensor with aligned modes.
    %
    %   A tensor_aligned is a tensor that also tracks the x-values for
    %   each mode. This is especially important for tensors where we 
    %   use just a subset of the indices in each mode. It can also be 
    %   used to remap the indices to any arbitrary set of values, including
    %   non-integer values.
    %
    %   Additionally, this class includes routines for downsampling in
    %   each mode, which can be used to reduce the size of the tensor.
    %
    %   See also TENSOR_UNALIGNED, KTENSOR_HIFI, CP_ALS_HIFI.

    % Code by Tamara Kolda, 2025. 
    
    properties
        xvals % x-values for each mode, usually 1:size(X,1)
    end

    methods

        function obj = tensor_aligned(X,xvals)
            %TENSOR_ALIGNED Constructor for the tensor_align class.
            %
            %   Y = TENSOR_ALIGNED(X) creates a tensor_aligned object 
            %   from the tensor X. It also sets the corresponding x-values to
            %   be 1:size(X,k) for each mode k.
            %
            %   Y = TENSOR_ALIGNED(X,XVALS) further specifies the x-values
            %   for each mode. XVALS is a cell array of length ndims(X) such
            %   that XVALS{k} is a row vector of strictly increasing values.
            %
            %   See also SUBSET_MODE, DOWNSAMPLE_MODE.

            % Check first argument
            if nargin == 0
                X = [];
            elseif ~isa(X,'tensor')
                error('X must be a tensor');
            end

            % Call tensor constructor
            obj = obj@tensor(X);

            % Set the x-values
            if nargin == 1 && isa(X,'tensor_aligned')
                obj.xvals = X.xvals;
            elseif  nargin == 2
                obj = set_xvals(obj,xvals);
            else
                obj.xvals = cell(1,ndims(obj));
                for k = 1:ndims(obj)
                    obj.xvals{k} = 1:size(obj,k);
                end
            end

        end % Constructor

        function obj = set_xvals(obj,xvals)
            % SET_XVALS Set the x-values for the tensor_aligned object.

            if length(xvals) ~= ndims(obj)
                error('Length of xvals must equal ndims(obj)');
            end
            for k = 1:ndims(obj)
                obj = set_mode_xvals(obj,k,xvals{k});
            end           

        end % Function set_xvals

        function obj = set_mode_xvals(obj,k,xvals)
            % SET_MODE_XVALS Set x-values for mode of tensor_aligned object.

            if length(xvals) ~= size(obj,k)
                error('Length of xvals must equal size(obj,k)');
            end
            if ~isvector(xvals) || any(diff(xvals) <= 0)
                error('xvals must be a row vector of strictly increasing values');
            end
            if ~isrow(xvals)
                xvals = xvals';
            end

            obj.xvals{k} = xvals;

        end % Function set_mode_xvals

        function obj = subset_mode(obj,k,idx,rescale)
            % SUBSET_MODE Subset a mode of the tensor.
            %
            %   TA = SUBSET_MODE(TA,K,IDX) returns a tensor_aligned object TA
            %   that is a subset of TA in mode K. IDX is a row vector of
            %   indices to keep. The resulting tensor is rescaled so that the
            %   norm of the tensor is the same as the original tensor.
            %
            %   TA = SUBSET_MODE(TA,K,IDX,RESCALE) allows the user to specify
            %   whether to rescale the tensor. The default is true.

            if ~isvector(idx) || any(diff(idx) <= 0)
                error('idx must be a strictly increasing row vector');
            end
            if ~isrow(idx)
                idx = idx';
            end
            if nargin < 4 
                rescale = true;
            end
            if rescale
                oldnorm = norm(obj);               
            end
            
            subs = repmat({':'},1,ndims(obj));
            subs{k} = idx;
            obj.data = obj.data(subs{:});
            obj.size(k) = numel(idx);
            obj.xvals{k} = obj.xvals{k}(idx);

            if rescale
                newnorm = norm(obj);
                obj = obj * (oldnorm / newnorm);
            end

        end % Function subset_mode

        function obj = downsample_mode(obj,k,npts,type,fuzzfactor)
            % DOWNSAMPLE_MODE Downsample a mode of the tensor.
            %
            %   TA = DOWNSAMPLE_MODE(TA,K,NPTS) returns a tensor_aligned object TA
            %   that is a downsampled version of TA in mode K. NPTS is the number
            %   of points to keep. The resulting tensor is rescaled so that the
            %   norm of the tensor is the same as the original tensor.
            %
            %   TA = DOWNSAMPLE_MODE(TA,K,NPTS,TYPE) allows the user to specify
            %   the type of downsampling. The options are:
            %      * 'random' - randomly select NPTS indices
            %      * 'linspace' - linearly spaced indices (rounded to integers)
            %      * 'nlinspace' (default) - nearly linearly spaced indices
            %
            %   TA = DOWNSAMPLE_MODE(TA,K,NPTS,TYPE,FACTOR) allows the user to
            %   specify the fuzz factor for the 'nlinspace' option. If the 
            %   minimum distance between two linearly spaced indices is MINSTEP,
            %   then the indices will be perturbed by up to FLOOR((MINSTEP-1)/FACTOR).
            %   The default is 2.5. For example, if the minimum distance between 
            %   linearly spaced indices is 10, then the indices will be perturbed 
            %   by up to 3 units in either direction.
            

            n = size(obj,k);

            if nargin < 4
                type = 'nlinspace';
            end

            if nargin < 5
                fuzzfactor = 2.5;
            end

            switch lower(type)
                case 'linspace'
                    idx = round(linspace(1,n,npts));
                case 'nlinspace'
                    idx = round(linspace(1,n,npts));
                    minstep = min(idx(2:end)-idx(1:end-1))-1;
                    distance = floor( minstep / fuzzfactor);
                    fudge = randi([-distance distance],1,npts-2);
                    idx(2:end-1) = idx(2:end-1) + fudge;
                otherwise % 'random'
                    idx = sort(randperm(n,npts),'ascend');
            end

            % Create the new tensor
            obj = subset_mode(obj,k,idx);

        end % Function downsample_mode

        function s = saveobj(obj)
            s = struct;
            s.data = obj.data;
            s.size = obj.size;
            s.xvals = obj.xvals;
        end 
        end % Methods

        methods (Static)
        function obj = loadobj(s)
            % LOADOBJ Custom load method for tensor_aligned.
            if isstruct(s)
                X = tensor(s.data,s.size);
                obj = tensor_aligned(X,s.xvals);           
            else
                obj = s;
            end
        end % Function loadobj

    end % Methods
end % Classdef