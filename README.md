# CP-HIFI Code for MATLAB

The code in this repository implements the CP-HIFI method for computing the CP decomposition of tensors whose modes are a hybrid of infinite- and finite-dimensional (HIFI). 
Here, we assume that we have some observed data corresponding to underlying continuous processes, such as fluid flow.
The CP factors corresponding to finite-dimensional modes are represented as vectors and those corresponding to infinfite-dimensional modes are represented as functions in an RKHS space. 
The code is written in MATLAB and relies on the [Tensor Toolbox for MATLAB](https://gitlab.com/tensors/tensor_toolbox) .

This work is based on a series of papers. The CP-HIFI framework comes from [1,2]. The implementations are detailed in [3].

* [1] R. Tang, T. Kolda, A. R. Zhang. Tensor Decomposition with Unaligned Observations. SIAM Journal on Matrix Analysis and Applications, 2026. https://doi.org/10.1137/24M1692836
* [2] B. W. Larsen, T. G. Kolda, A. R. Zhang, A. H. Williams. Tensor Decomposition Meets RKHS: Efficient Algorithms for Smooth and Misaligned Data. arXiv:2408.05677, 2024. http://arxiv.org/abs/2408.05677
* [3] Johannes Brust and Tamara G. Kolda, Fast and Accurate CP-HIFI Tensor Decompositions, in preparation, 2026.

## Setup

* Download the [Tensor Toolbox for MATLAB](https://gitlab.com/tensors/tensor_toolbox) and add its directory to your MATLAB path.
* Clone this repository and add it to your MATLAB path.

## Special Classes

* `tensor_aligned` - Derived from `tensor` class in the Tensor Toolbox but also tracks the x-values for each mode. 
  * Easy to down-sample via its constructor or the `downsample_mode` method.
  * It can also be used to track when the modes do not correspond to evenly spaced x-values or if we want special x-values for HIFI. For instance, the x-values may all lie in the interval [0,1], which is a common assumption for the kernel functions. These can be specified during construction or see the `set_mode_vals` method.
  * To see methods specific to this class, type `setdiff(methods('tensor_aligned'),methods('tensor'))`.

* `tensor_unaligned` - Derived from `sptensor` class in the Tensor Toolbox but also tracks the x-values for each mode. 
  * Analogous to `tensor`/`tensor_aligned`. 
  * Stores an **incomplete** tensor, meaning that only a subset of the entries are observed. 
  * Its constructor can downsample from a `tensor` or `tensor_aligned` object. 
  * To see methods specific to this class type `setdiff(methods('tensor_unaligned'),methods('sptensor'))`.

* `ktensor_hifi` - Derived from `ktensor` class in the Tensor Toolbox. This class does a few things of important note:
  *  It tracks the x-values for each mode and extends the ktensor/viz function to automatically use those x-values. Even though HIFI is in its name, it can be used for this feature alone.
  * It is the output of `cp_als_hifi`. It holds a standard ktensor but also the ability to resample any infinite-dimensional mode to a different number of points because it has saved the kernel function and weight matrix.
  * To see the methods specific to this class, type `setdiff(methods('ktensor_hifi'),methods('ktensor'))`. 

## Methods

* `cp_als_hifi` - Compute the CP decomposition of a tensor with possible HIFI modes. If none of the modes are HIFI, this should produce the same results as `cp_als`. Unlike `cp_als`, however, this method also has a `nonneg` option and uses `lsqnonneg` in MATLAB. It can take a `tensor`, `tensor_aligned`, or `tensor_unaligned` as input. It uses the following helper methods:
  - `cp_als_hifi_solver_af` (AF = Aligned Finite)
  - `cp_als_hifi_solver_uf` (UF = Unaligned Finite)
  - `cp_als_hifi_solver_ai` (AI = Aligned Infinite)
  - `cp_als_hifi_solver_ui` (UI = Unaligned Infinite)

* `kernfunc_gaussian` - Gaussian kernel function, for use with `ktensor_hifi`.
