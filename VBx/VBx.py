#!/usr/bin/env python

# Copyright 2021 Lukas Burget, Mireia Diez (burget@fit.vutbr.cz, mireia@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Revision History
#   L. Burget   20/1/2021 1:00AM - original version derived from the more
#                                  complex VB_diarization.py avaiable at
# https://github.com/BUTSpeechFIT/VBx/blob/e39af548bb41143a7136d08310765746192e34da/VBx/VB_diarization.py
#

import numpy as np
from scipy.special import logsumexp


def VBx(X, Phi, loopProb=0.9, Fa=1.0, Fb=1.0, pi=10, gamma=None, maxIters=10,
        epsilon=1e-4, alphaQInit=1.0, ref=None, plot=False,
        return_model=False, alpha=None, invL=None):

    """
    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors
                  (e.g. x-vectors) for T frames
    X: T×D 的矩阵，每列是一个 D 维特征向量（例如x-vectors），代表 T 帧。
    Phi         - D array with across-class covariance matrix diagonal.
                  The model assumes zero mean, diagonal across-class and
                  identity within-class covariance matrix.
    Phi: 长度为 D 的数组，表示跨类协方差矩阵的对角线。
    loopProb    - Probability of not switching speakers between frames
    loopProb: 帧间不切换说话人的概率。
    Fa          - Scale sufficient statiscits
    Fb          - Speaker regularization coefficient Fb controls the final number of speakers
    Fa, Fb: VB-HMM的参数，控制统计量和说话人正则化。
    pi          - If integer value, it sets the maximum number of speakers
                  that can be found in the utterance.
                  If vector, it is the initialization for speaker priors (see Outputs: pi)
    pi: 如果是整数，表示可能的最大说话人数；如果是向量，表示初始化的说话人先验概率。
    gamma       - An initialization for the matrix of responsibilities (see Outputs: gamma)
    gamma: 初始化的责任矩阵（每帧属于某个说话人的后验概率）。
    maxIters    - The maximum number of VB iterations
    maxIters: 最大迭代次数。
    epsilon     - Stop iterating, if the obj. fun. improvement is less than epsilon
    epsilon: 迭代停止条件，当目标函数的改进小于该值时停止。
    alphaQInit  - Dirichlet concentraion parameter for initializing gamma
    alphaQInit: 初始化 gamma 的Dirichlet集中参数。
    ref         - T dim. integer vector with per frame reference speaker IDs (0:maxSpeakers)
    ref: 参考说话人ID，用于计算Diarization Error Rate（DER）。
    plot        - If set to True, plot per-frame marginal speaker posteriors 'gamma'
    plot: 绘制每帧说话人后验概率的图形。
    return_model- Return also speaker model parameter
    return_model: 返回说话人模型参数。
    alpha, invL - If provided, these are speaker model parameters used in the first iteration
    alpha, invL: 如果提供，这些是用于第一次迭代的说话人模型参数。

    Outputs:
    gamma       - S x T matrix of responsibilities (marginal posteriors)
                  attributing each frame to one of S possible speakers
                  (S is defined by input parameter pi)
    gamma: S×T的矩阵，表示每帧属于 S 个可能说话人的后验概率。    
    pi          - S dimensional column vector of ML learned speaker priors.
                  This allows us to estimate the number of speaker in the
                  utterance as the probabilities of the redundant speaker
                  converge to zero.
    pi: 长度为 S 的向量，表示最大似然估计的说话人先验概率。
    Li          - Values of auxiliary function (and DER and frame cross-entropy
                  between gamma and reference, if 'ref' is provided) over iterations.
    Li: 辅助函数值的列表。
    alpha, invL - Speaker model parameters returned only if return_model=True
    alpha, invL: 说话人模型参数（仅在 return_model 为 True 时返回）。

    Reference:
      Landini F., Profant J., Diez M., Burget L.: Bayesian HMM clustering of
      x-vector sequences (VBx) in speaker diarization: theory, implementation
      and analysis on standard tasks
    """
    """
    The comments in the code refers to the equations from the paper above. Also
    the names of variables try to be consistent with the symbols in the paper.
    """

    #初始化
    #确定特征维度 D。
    D = X.shape[1]  # feature (e.g. x-vector) dimensionality
    # 初始化说话人先验概率 pi 和责任矩阵 gamma。
    if type(pi) is int:
        pi = np.ones(pi)/pi

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with
        # concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    assert(gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0])
    #预计算常数和矩阵
    #计算每帧的常数项 G
    #计算 Φ 的平方根 V 和ρ。

    G = -0.5*(np.sum(X**2, axis=1, keepdims=True) + D*np.log(2*np.pi))  # per-frame constant term in (23)
    V = np.sqrt(Phi)  # between (5) and (6)
    rho = X * V  # (18)
    Li = []
    #主要迭代过程
    #计算说话人模型参数 invL 和 alpha。
    #计算每帧的对数概率 log_p_。
    #调用 forward_backward 函数计算责任矩阵 gamma 和其他参数。
    #计算ELBO（证据下限）。
    #更新说话人先验概率 pi。
    #保存ELBO值。
    for ii in range(maxIters):
        # Do not start with estimating speaker models if those are provided
        # in the argument
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa/Fb * gamma.sum(axis=0, keepdims=True).T*Phi)  # (17) for all speakers
            alpha = Fa/Fb * invL * gamma.T.dot(rho)  # (16) for all speakers
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL+alpha**2).dot(Phi) + G)  # (23) for all speakers
        tr = np.eye(len(pi)) * loopProb + (1-loopProb) * pi  # (1) transition probability matrix
        gamma, log_pX_, logA, logB = forward_backward(log_p_, tr, pi)  # (19) gamma, (20) logA, (21) logB, (22) log_pX_
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)  # (25)
        pi = gamma[0] + (1-loopProb)*pi * np.sum(np.exp(logsumexp(
            logA[:-1], axis=1, keepdims=True) + log_p_[1:] + logB[1:] - log_pX_
        ), axis=0)  # (24)
        pi = pi / pi.sum()
        Li.append([ELBO])

        # if reference is provided, report DER, cross-entropy and plot figures
        # 如果提供参考标签 ref，计算DER和交叉熵。
        if ref is not None:
            Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]
            #如果 plot 为 True，绘制每帧的说话人后验概率图。
            if plot:
                import matplotlib.pyplot
                if ii == 0:
                    matplotlib.pyplot.clf()
                matplotlib.pyplot.subplot(maxIters, 1, ii+1)
                matplotlib.pyplot.plot(gamma, lw=2)
                matplotlib.pyplot.imshow(np.atleast_2d(ref),
                                         interpolation='none', aspect='auto',
                                         cmap=matplotlib.pyplot.cm.Pastel1,
                                         extent=(0, len(ref), -0.05, 1.05))
        #检查迭代停止条件并返回结果。
        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            if ELBO - Li[-2][0] < 0:
                print('WARNING: Value of auxiliary function has decreased!')
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())


# Calculates Diarization Error Rate (DER) or per-frame cross-entropy between
# reference (vector of per-frame zero based integer speaker IDs) and gamma
# (per-frame speaker posteriors). If expected=False, gamma is converted into
# hard labels before calculating DER. If expected=TRUE, posteriors in gamma
# are used to calculated "expected" DER.
#计算分割错误率（DER）或每帧参考说话人ID和后验概率 gamma 之间的交叉熵。
def DER(q, ref, expected=True, xentropy=False):
    from scipy.sparse import coo_matrix
    from scipy.optimize import linear_sum_assignment
    if not expected:  # replce probabilities in q by zeros and ones
        q = coo_matrix((np.ones(len(q)), (range(len(q)), q.argmax(1)))).toarray()

    ref_mx = coo_matrix((np.ones(len(ref)), (range(len(ref)), ref)))
    err_mx = ref_mx.T.dot(-np.log(q+np.nextafter(0, 1)) if xentropy else -q)
    min_cost = err_mx[linear_sum_assignment(err_mx)].sum()
    return min_cost/float(len(ref)) if xentropy else (len(ref) + min_cost)/float(len(ref))


#实现前向后向算法，用于计算每帧状态占用后验概率 pi。
def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. starting in the state)
    Outputs:
        pi  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    eps = 1e-8
    ltr = np.log(tr + eps)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + eps)
    lbw[-1] = 0.0

    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii-1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls)-1)):
        lbw[ii] = logsumexp(ltr + lls[ii+1] + lbw[ii+1], axis=1)

    tll = logsumexp(lfw[-1], axis=0)
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw
