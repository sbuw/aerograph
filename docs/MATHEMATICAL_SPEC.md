# Mathematical Specification

## 1. Graph Construction & Correlation Matrix
For each grid node $i$, center and standardize the pressure anomaly time series $x_i(t)$ of length $T$:
$$\tilde{x}_i(t) = \frac{x_i(t) - \bar{x}_i}{\sigma_i}$$

Compute the cross-correlation function across discrete lags $\tau \in [0, \tau_{\max}]$:
$$R_{ij}(\tau) = \frac{1}{T - \tau} \sum_{t=1}^{T - \tau} \tilde{x}_i(t) \tilde{x}_j(t + \tau)$$

Extract the peak absolute correlation:
$$r_{ij}^* = \max_\tau |R_{ij}(\tau)|$$

Construct the adjacency matrix $A$ by applying a threshold $\theta$ and pruning localized spatial neighbors:
$$A_{ij} = \begin{cases} r_{ij}^*, & \text{if } r_{ij}^* \ge \theta \text{ and } \text{dist}(i, j) > d_{\min} \\ 0, & \text{otherwise} \end{cases}$$
where $d_{\min}$ eliminates trivial adjacent-cell correlations ($A_{ii} = 0$).

## 2. Normalized Graph Laplacian
Degree matrix $D$:
$$D_{ii} = \sum_{j} A_{ij}$$

Symmetric normalized Laplacian $L$:
$$L = I - D^{-1/2} A D^{-1/2}$$

Eigen-decomposition:
$$L = U \Lambda U^T$$
where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$ ($0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2$), and columns of $U = [u_1, \dots, u_N]$ form an orthonormal basis of graph harmonics.

## 3. Spectral Graph Wavelets (SGWT)
Wavelet operator $T_g = g(sL)$ with scaling factor $s$ and band-pass filter kernel $g$:
$$\psi_{s, i} = g(sL) \delta_i = \sum_{k=1}^N g(s \lambda_k) u_k(i) u_k$$

To bypass full $O(N^3)$ diagonalization on large graphs, compute $g(sL)$ using truncated $M$-th order Chebyshev polynomials:
$$g(sL) \approx \frac{1}{2} c_0 I + \sum_{m=1}^M c_m T_m(\tilde{L})$$
where $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$, and $T_m$ satisfies recurrence $T_m(x) = 2x T_{m-1}(x) - T_{m-2}(x)$.