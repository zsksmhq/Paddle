# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numbers

import paddle 
from paddle.distribution import Beta
from paddle.distribution.distribution import Distribution
# from paddle.distribution.utils import broadcast_all


class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    """

    def __init__(self, dim, concentration=1.0):
        if dim < 2:
            raise ValueError(
                f"Expected dim to be an integer greater than or equal to 2. Found dim={dim}."
            )
        self.dim = dim
        [self.concentration] = self._to_tensor(concentration)
        batch_shape = self.concentration.shape
        event_shape = (dim, dim)
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = paddle.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
        )
        offset_zero = paddle.zeros((1,), offset.dtype)
        offset = paddle.concat([offset_zero, offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        super().__init__(batch_shape, event_shape)

    def sample(self, sample_shape=()):
        # This uses the Onion method, but there are a few differences from [1] Sec. 3.2:
        # - This vectorizes the for loop and also works for heterogeneous eta.
        # - Same algorithm generalizes to n=1.
        # - The procedure is simplified since we are sampling the cholesky factor of
        #   the correlation matrix instead of the correlation matrix itself. As such,
        #   we only need to generate `w`.
        breakpoint()
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = paddle.randn(
            self._extend_shape(sample_shape), dtype=y.dtype
        ).tril(-1)
        u_hypersphere = u_normal / paddle.linalg.norm(u_normal, axis=-1, keepdim=True) 
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = paddle.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        eps = paddle.finfo(w.dtype).tiny
        diag_elems = paddle.clip(1 - paddle.sum(w**2, axis=-1), min=eps).sqrt()
        w += paddle.nn.functional.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        diag_elems = value.diagonal(axis1=-1, axis2=-2)[..., 1:]
        order = paddle.arange(2, self.dim + 1)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = paddle.sum(order * diag_elems.log())
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = paddle.lgamma(alpha) * dm1
        numerator = paddle.multigammaln(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term
