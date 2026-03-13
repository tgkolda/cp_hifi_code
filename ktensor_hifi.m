classdef ktensor_hifi < ktensor
    %KTENSOR_HIFI Hybrid finite and infinite-dimensional (HIFI) Kruskal tensor.
    %
    %   This class extends the ktensor class to represent hybrid finite and
    %   inifinite dimensional modes. For the infinite-dimensional modes, it
    %   stores kernel functions and weight matrices used to represent each
    %   infinite-dimensional mode. Each infinite-dimensional mode is also has a
    %   corresponding finite-dimensional factor matrix evaluated at a set of
    %   x-values. Any infinite-dimensional mode can be resampled. 
    % 
    %   Every mode store x-values corresponding to the finite-dimensional
    %   evaluation points. This can be useful whenever these samples are not
    %   evenly spaced, even if no mode is inifitie-dimensional.
    %
    %   See also KTENSOR, TENSOR_ALIGNED, CP_ALS_HIFI.
    

    % Code by Tamara Kolda, 2025. 
    % Modification by Johannes Brust, 2025-2026.

    properties
        kernfunc, % kernel functions, one per mode (SHOULD NEVER CHANGE!)
        weightmat, % weight matrices, one per mode (SHOULD NEVER CHANGE!)
        dvals, % design points used to create weight matrix (SHOULD NEVER CHANGE!)
        xvals, % xvals for current kernel and factor matrices
    end
    methods

        function obj = ktensor_hifi(lambda,U,xvals,kernfunc,weightmat,dvals)
            %KTENSOR_HIFI Constructor.
            %
            %   KT = KTENSOR_HIFI(LAMBDA, U, XVALS, KERNFUNC, WEIGHTMAT) constructs a
            %   ktensor_hifi object. LAMBDA is a vector of weights, U is a cell array
            %   of factor matrices, XVALS is a cell array of design points used to
            %   create the weight matrices, KERNFUNC is a cell array of kernel
            %   functions, and WEIGHTMAT is a cell array of the corresponding weight
            %   matrices. All the inputs are cell arrays, which are empty except for
            %   HIFI modes. If KERNFUNC{K} is empty, then mode K is a regular discrete
            %   mode.       

            % Error check on lambda
            if ~isvector(lambda)
                error('lambda must be a vector');
            end

            % Extract number of dimensions
            %d = length(lambda);
            d = length(U);

            % Error check on U
            if ~iscell(U) % || length(U) ~= d
                error('U must be a cell array with %d cells', d);
            end
            
            % Checks on xvals
            if nargin < 3
                xvals = cell(d,1);
            elseif ~iscell(xvals) || length(xvals) ~= d
                error('xvals must be a cell array with %d cells', d);
            end

            % Check on kernel functions
            if nargin < 4
                kernfunc = cell(d,1);
            elseif ~iscell(kernfunc) || length(kernfunc) ~= d
                error('kernfunc must be a cell array with %d cells', d);
            end

            % Check on weight matrices
            if nargin < 5
                weightmat = cell(d,1);
            elseif ~iscell(weightmat) || length(weightmat) ~= d
                error('weightmat must be a cell array with %d cells', d);
            end

            % Check on design points
            if nargin < 6
                dvals = xvals;
            elseif ~iscell(dvals) || length(dvals) ~= d
                error('dvals must be a cell array with %d cells', d);
            end
            
            for k = 1:d
                if isempty(kernfunc{k}) % finite-dimensional mode
                    if isempty(U{k})
                        error('Factor matrix %d is empty but there is no kernel function', k);
                    end
                    if ~isempty(weightmat{k})
                        warning('Mode %d has a specified weight matrix but no kernel function', k);
                    end
                    if isempty(xvals{k})
                        xvals{k} = 1:size(U{k},1);
                    elseif ~isvector(xvals{k})
                        error('xvals must be a vector');
                    elseif ~isrow(xvals{k})
                        xvals{k} = xvals{k}';
                    end
                    if length(xvals{k}) ~= size(U{k},1)
                        error('Size mismatch in xvals and U for mode %d',k);
                    end
                else % infinite-dimensional mode
                    if ~isa(kernfunc{k}, 'function_handle')
                        error('kernfunc{%d} must be empty or a function handle', k);
                    end
                    if isempty(weightmat{k})
                        error('Mode %d has a kernel function but no weight matrix', k);
                    end
                    if isempty(xvals{k})
                        error('Mode %d has a kernel function but no x-values',k)
                    elseif ~isvector(xvals{k})
                        error('xvals must be a vector');
                    elseif ~isrow(xvals{k})
                        xvals{k} = xvals{k}';
                    end
                    if isempty(dvals{k})
                        dvals{k} = xvals{k};
                    elseif ~isvector(dvals{k})
                        error('dvals must be a vector');
                    elseif ~isrow(dvals{k})
                        dvals{k} = dvals{k}';
                    end
                    % Fill in U{k} or check consistency
                    Utmp = (kernfunc{k}((xvals{k})',dvals{k}))*weightmat{k};
                    if isempty(U{k})
                        U{k} = Utmp;
                    else
                        if ~isequal(size(U{k}), size(Utmp))
                            error('Size inconsistency in factor matrix %d', k);
                        end
                        if norm(U{k}-Utmp) > 1e-4 %sqrt(eps) %10*eps
                            error('Value inconsistency in factor matrix %d',k);
                        end
                    end
                end
            end

            obj = obj@ktensor(lambda,U);
            obj.kernfunc = kernfunc;
            obj.weightmat = weightmat;
            obj.dvals = dvals;
            obj.xvals = xvals;

        end

        function tf = is_hifimode(obj,k)
            %IS_HIFIMODE Check if mode k is an HIFI mode.
            tf = ~isempty(obj.kernfunc{k});
        end

        function range = mode_range(obj,k)
            %GET_RANGE Get the range of the dvals for mode k.
            %
            %   [MIN,MAX] = GET_RANGE(K) returns the minimum and maximum
            %   values of the design points for mode K.
            range = [min(obj.dvals{k}),max(obj.dvals{k})];
        end

        function TA = full(obj)
            %FULL Convert ktensor_hifi to a full tensor_aligned object.

            X = full@ktensor(obj);
            TA = tensor_aligned(X);
            for k = 1:ndims(obj)
                if is_hifimode(obj,k)
                    TA = set_mode_xvals(TA,k,obj.xvals{k});
                end
            end
        end

        function obj = resample_modes(obj,sz)
            %RESAMPLE_MODES Resample all HIFI modes of the tensor.
            %
            %   KT = RESAMPLE_MODES(KT,SZ) resamples HIFI mode K of KT
            %   using SZ{K} points if SZ{K} > 0.
            %   See RESAMPLE_MODE for more details.

            d = ndims(obj);
            for k = 1:d
                if is_hifimode(obj,k)
                    obj = resample_mode(obj,k,sz(k));
                end
            end
        end

        function obj = resample_mode(obj,k,npts,range,reweight)
            %RESAMPLE_MODE Resample a mode of the tensor.
            %
            %   KT = RESAMPLE_MODE(KT,K,N) resamples the kth mode of KT
            %   using N points. By default, the range is MODE_RANGE(KT,K).
            %   The new factor matrix is rescaled so that the column norms
            %   are the same as the corresponding column norms in the old
            %   factor matrix.
            %
            %   KT = RESAMPLE_MODE(KT,K,N,RANGE) specifies the range as [A,B].

            if nargin < 3
                error('Must specify mode and number of points');
            end

            if isempty(obj.kernfunc{k})
                error('Mode %d is not an HIFI mode',k);
            end

            % Default range is the range of the mode's dvals
            if nargin < 4 || isempty(range)
                range = mode_range(obj,k);
            end

            if nargin < 5 
                reweight = true;
            end

            % Compute new xvals
            xx = linspace(range(1),range(2),npts);

            % Reset the x-values
            obj = reset_mode_xvals(obj,k,xx,reweight);
        end

        function obj = reset_mode_xvals(obj,k,xx,reweight)
            %RESET_MODE_XVALS Reset the x-values for mode k.
            %
            %   KT = RESET_MODE_XVALS(KT,K,XX) resets the x-values for 
            %   mode K to XX. If mode K is a continuous mode, then the 
            %   new values are computed using the continuous function that 
            %   defines that mode. Otherwise, XX must be a subset of the
            %   current x-values for mode K, and these are "extracted" from
            %   the current factor matrix. 
            %
            %   KT = RESET_MODE_XVALS(KT,K,XX,false) prevents the default
            %   behavior of rescaling the new factors so that their norms
            %   match the old factors (preserving the overall norm of the
            %   ktensor).

            if nargin < 4
                reweight = true;
            end

            if is_hifimode(obj,k)
                % Extract info from object for mode k
                W = obj.weightmat{k};
                kfunc = obj.kernfunc{k};
                dd = obj.dvals{k};

                % Compute new kfunc matrix
                K = kfunc(xx',dd);

                % Compute new factor matrix
                A = K*W;

                % Reweight so that each column has the same norm as
                % corresponding column in old factor matrix
                if reweight
                    for j = 1:size(A,2)
                        reweight = norm(obj.u{k}(:,j))/norm(A(:,j));
                        A(:,j) = reweight * A(:,j);
                    end
                end

                % Update kth mode
                obj.u{k} = A;
                obj.xvals{k} = xx;

            else
                [tf,idx] = ismember(xx,obj.xvals{k});
                if any(~tf)
                    error('xx must be a subset of the obj.xvals{k}');
                end

                % Extract the new factor matrix
                A = obj.u{k}(idx,:);

                % Rescale the columns to match the old
                if reweight
                    for j = 1:size(A,2)
                        reweight = norm(obj.u{k}(:,j))/norm(A(:,j));
                        A(:,j) = reweight * A(:,j);
                    end
                end

                obj.xvals{k} = obj.xvals{k}(idx);
                obj.u{k} = A;
            end
        end

        function varargout = viz(obj,varargin)
            % VIZ Visualize the ktensor using HIFI points.
            %
            % [...] = VIZ(K) plots the ktensor_hifi K adding an extra
            % argument to specify the x-values for each mode based on
            % the current HIFI x-values.
            if nargin == 1
                varargin = {};
            end
            varargout = cell(1,nargout);
            [varargout{:}] = viz@ktensor(obj,varargin{:},'Xvals',obj.xvals);
        end

        function obj = normalize(obj,arg2,arg3)
            %NORMALIZE Normalize the factor matrices of a ktensor.
            %
            %   KT = NORMALIZE(KT) normalizes the columns of the factor
            %   matrices of a ktensor so that the columns have unit length,
            %   resulting in new lambda values. The weight matrices of HIFI
            %   modes are also updated accordingly.

            tf_arrange = false;
            tf_reabsorb = false;
            tf_error = false;

            if nargin > 3
                tf_error = true;
            elseif nargin == 2
                if isequal(arg2,'sort')
                    tf_arrange = true;
                elseif arg2 == 0
                    tf_reabsorb = true;
                elseif arg2 ~=2
                    tf_error = true;
                end
            elseif nargin == 3
                if isequal(arg2,'sort')
                    tf_arrange = true;
                else
                    tf_error = true;
                end
                if arg3 == 0
                    tf_reabsorb = true;
                elseif arg3 ~= 2
                    tf_error = true;
                end
            end

            if tf_error
                error('Requested normalization is not yet available for ktensor_hifi')
            end

            if tf_arrange
                obj = arrange(obj);
            else
                %TODO: replace with vecnorm
                d = ndims(obj);
                r = length(obj.lambda);
                for j = 1:r
                    for k = 1:d
                        tmp = norm(obj.u{k}(:,j));
                        if (tmp > 0)
                            obj.u{k}(:,j) = obj.u{k}(:,j) / tmp;
                            if ~isempty(obj.kernfunc{k})
                                obj.weightmat{k}(:,j) = obj.weightmat{k}(:,j) / tmp;
                            end
                        end
                        obj.lambda(j) = obj.lambda(j) * tmp;
                    end
                end
            end

            if tf_reabsorb
                % redistribute the weight
                D = diag(nthroot(obj.lambda,ndims(obj)));
                for k = 1:ndims(obj)
                    obj.u{k} = obj.u{k} * D;
                    if ~isempty(obj.kernfunc{k})
                        obj.weightmat{k} = obj.weightmat{k} * D;
                    end
                end
                obj.lambda = ones(size(obj.lambda));
            end

        end

        function resid = norm_masked_diff(obj,X,nzchunksize)
            %NORM_MASKED_DIFF Compute difference norm with incomplete tensor.
            %
            %   D = NORM_MASKED_DIFF(KT,X) computes the Frobenius norm of the difference
            %   between the incomplete tensor X and KT, using only the known entries of X.

            %% Extract scarce data
            Xsubs = X.subs;
            Xvals = X.vals;
            d = ndims(X);
            r = ncomponents(obj);

            %% Chunk the nonzeros
            nz = size(Xsubs,1);
            if nargin < 3
                nzchunksize = 5000;
            end
            cstart= 1:nzchunksize:nz;
            cend = [cstart(2:end)-1 nz];
            nchunks = length(cstart);
            csz = cend-cstart+1;

            %% Compute the residual squared
            residsqr = 0;
            for c = 1:nchunks
                chunk = cstart(c):cend(c);
                Hchunk = ones(csz(c),r);
                for k = 1:d
                    Akexp = obj.u{k}(Xsubs(chunk,k),:);
                    Hchunk = Hchunk .* Akexp;
                end
                Mvals = Hchunk * obj.lambda;
                residsqr = residsqr + sum((Xvals(chunk) - Mvals).^2);
            end

            %% Take the square root
            resid = sqrt(residsqr);

        end


        function [obj,p] = arrange(obj,foo)
            %ARRANGE Arranges the rank-1 components of a ktensor_hifi.
            %
            %   X = ARRANGE(X) normalizes the columns of the factor matrices and then
            %   sorts the ktensor components by magnitude, greatest to least.
            %
            %   X = ARRANGE(X,N) absorbs the weights into the Nth factor matrix instead
            %   of lambda.
            %
            %   X = ARRANGE(X,P) rearranges the components of X according to the
            %   permutation P. P should be a permutation of 1 to NCOMPONENTS(X).
            %
            %   [X,P] = ARRANGE(...) returns also the permutation of the components.
            %
            %   Examples
            %   K = ktensor([3; 2], rand(4,2), rand(5,2), rand(3,2))
            %   arrange(K) %<--Normalize and sort according to weight vector
            %   arrange(K,[2, 1]) %<--Order components according to permutation
            %
            %   See also KTENSOR, NCOMPONENTS, NORMALIZE.
            %
            %Tensor Toolbox for MATLAB: <a href="https://www.tensortoolbox.org">www.tensortoolbox.org</a>

            % Just rearrange and return if second argument is a permutation
            if exist('foo','var') && (length(foo) > 1)
                p = foo; % save the permutation
                obj.lambda = obj.lambda(p);
                for i = 1 : ndims(obj)
                    obj.u{i} =obj.u{i}(:,p);
                    if ~isempty(obj.kernfunc{i})
                        obj.weightmat{i} = obj.weightmat{i}(:,p);
                    end
                end
                return;
            end

            % Normalize factor matrices
            obj = normalize(obj);

            % Sort lambda in decreasing order
            [obj.lambda,p] = sort(obj.lambda,'descend');
            for k = 1:ndims(obj)
                obj.u{k} = obj.u{k}(:,p);
                if ~isempty(obj.kernfunc{k})
                    obj.weightmat{k} = obj.weightmat{k}(:,p);
                end
            end

            % Absorb the weight into one factor, if requested
            if exist('foo','var')
                r = length(obj.lambda);
                D = full(spdiags(obj.lambda,0,r,r));
                obj.u{foo} = obj.u{foo} * D;
                if ~isempty(obj.kernfunc{foo})
                    obj.weightmat{foo} = obj.weightmat{foo} * D;
                end
                obj.lambda = ones(size(obj.lambda));
            end
        end

        function obj = fixsigns(obj)
            %FIXSIGNS Fix the signs of the factor matrices of a ktensor.
            %
            %   KT = FIXSIGNS(KT) fixes the signs of the factor matrices of a ktensor 
            %   so that the largest element in each column is positive. This also 
            %   updates the weight matrices of HIFI modes accordingly.

            d = ndims(obj);
            KTnew = fixsigns@ktensor(obj);
            for k = 1:d
                if ~isempty(obj.kernfunc{k})
                    tmp = KTnew.u{k} ./ obj.u{k}; % Track sign flips
                    s = max(tmp,[],1); % ignores NaNs in tmp
                    obj.weightmat{k} = obj.weightmat{k} .* s; % Apply sign flips
                end
            end
            obj.lambda = KTnew.lambda;
            obj.u = KTnew.u;
        end

        function s = saveobj(obj)
            s.lambda = obj.lambda;
            s.u = obj.u;
            s.xvals = obj.xvals;
            s.kernfunc = obj.kernfunc;
            s.weightmat = obj.weightmat;
            s.dvals = obj.dvals;
        end

    end % METHODS

    methods (Static)

        function obj = loadobj(s)

            if isstruct(s)
                obj = ktensor_hifi(s.lambda,s.u,s.xvals,s.kernfunc,s.weightmat,s.dvals);
            else
                obj = s;
            end

        end

    end
end % CLASSDEF