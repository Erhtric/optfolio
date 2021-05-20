

# 1. Portfolio Optimization 

From *Advanced Methods in Portfolio Optimization for Trading Strategies and Smart Beta, J. de La Batut, 2018*

## 1.1 Mathematical Framework

### 1.1.1 Convex Program and optimization

Assume that *all the weights are positive*. Then, this sum is a **convex** **sum**. 

**Definition 1.1**: $f: X \rightarrow R$ is a convex function if and only if $\forall x_1x_2 \in X, \forall t \in [0,1]: f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)$.

We could define the following general optimization problem for the function *f*
$$
\begin{align}
	\min \quad & f(x)\\
	\textrm{s.t.} \quad & g_i(x) \leq 0 \quad\forall i \leq p \nonumber\\
  	& h_j(x) = 0 \quad\forall j \leq m \nonumber\\
\end{align}
$$
where $g$ are the inequality constraints and $h$ the equality ones. According to Kuhn and Tucker, $w^*$ is a solution of the problem if and only if the following hold:

- *Stationarity*: $\nabla\mathcal{L}(w^*) = 0$,
- *Primal feasibility*: $w^*$ is feasible,
- *Dual feasibility*: all the Lagrange multipliers for inequality constraints $\gamma_i \geq 0$ or null. 
- *Complementary slackness*: $\gamma_ig_i(w^*) = 0$.

where the Lagrangian is defined as usual:
$$
\mathcal{L}(w) = f(w) + \sum_{i=1}^p \gamma_ig_i(w)  + \sum_{i=1}^m \mu_ih_i(w)
$$

When the objective function *f* is quadratic, the optimization problem will be:
$$
\begin{align}
	\min \quad & \dfrac{1}{2}w^TPw \,+ q^Tw \\
	\textrm{s.t.} \quad & Gw \leq h \nonumber\\
  	& Aw = b \nonumber\\
\end{align}
$$
where $w \in \mathbb{R}^n$, $G \in \mathbb{R}^{n \times m}$, $h \in \mathbb{R}^m$, $A\in \mathbb{R}^{p \times n}$ and $b \in \mathbb{R}^p$ (there are *m* inequality constraints and *p* equality constraints).

### 1.1.2 Convex Vector Optimization

A set $S \subseteq \mathbb{R}^n$ is a convex set if it contains all line segments joining any pair of points in *S*, 
$$
x, y \in S, \theta > 0 \Rightarrow \theta x + (1-\theta)y \in S
$$
A function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is convex if its domain is convex and for all points $x,y$ belonging to the domain and $\theta \in [0,1]$ holds
$$
F(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)
$$
A function $f$ is concave if it is convex. Geometrically speaking, one can think of the curve of a convex function as always lying below the line segment of any two points.

<img src="C:\Users\Eric R\AppData\Roaming\Typora\typora-user-images\image-20210422102409276.png" alt="image-20210422102409276" style="zoom: 67%;" />

A vector convex optimization has standard form:
$$
\begin{aligned}
        \min_{w} \quad & f_0(w)\\
        \textrm{s.t.} \quad & g_i(w) \leq 0, \quad i=1,\dots,m\\
        & h_j(w) = 0, \quad j=1,\dots,p
\end{aligned}
$$
Here, $f_0(w)$ is the convex o.f., $g_i(w)$ are the convex inequality constraint functions and $h_j(w)$ are the equality constraint functions which can be expressed in linear form $Aw + B$.

### 1.1.3 Multi-objective optimization

The multi-objective approach combines multiple objective functions $f_1(w), f_2(w), \dots, f_m(w)$  into one objective function by assigning a weighting coefficient to each objective. The standard solution technique is to minimize a positively weighted convex sum of the objectives using single-objective method, that is:
$$
\begin{aligned}
	\min \quad &F(w) = \sum_{i=1}^N a_if_i(w), \quad a_i>0, i\in\{1,\dots,m\}\\
		\textrm{s.t.} \quad & g_i(w) \geq 0, \quad i=1,\dots,m\\
        & h_j(w) = 0, \quad j=1,\dots,p\\
        & w_i^{(L)} \leq w_i\leq w_i^{(U)} \quad i = 1,\dots,k 
\end{aligned}
$$
The weight of an objective is chosen in proportion to the relative importance of the objective.

The concept of optimality in multiple-objective optimization is characterized by Pareto optimality. Essentially, a vector $w$ is said to be Pareto optimal if and only if there is no $w$ such that $f_i(w) \leq f_i(w^*)$ for all $i \in \{1, \dots, m\}$. In other words, $w^*$ is the Pareto point if $F(w^*)$ achieves its minimal value.

Since investors are interested in minimizing risk and maximizing expected return at the same time, the portfolio optimization problem can be treated as a multi-objective optimization problem. One can attain Pareto optimality in this case because the formulation of (6) belongs to the category of convex vector optimization, which guarantees that any local optimum is a global optimum.

#### 1.1.3.1 Solving multi-objective optimization

The multi-objective optimization can be solved using Lagrangian multiplier:
$$
\mathcal{L}(w) = -w^Tr+ \mu w^T\Sigma w + \lambda(\mathbf{1}^Tw -1)
$$
Set $\dfrac{\partial\mathcal{L}}{\partial w} = 0$, it follows that:
$$
w = \dfrac{1}{2\mu}\Sigma^{-1}(r - \lambda )
$$

To solve the Lagrangian multiplier $\lambda$, substitute equation (8) to the constraint $\mathbf{1}^Tw = 1$:
$$
\lambda = \dfrac{\mathbf{1}^T\Sigma^{-1}r}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}} - \dfrac{2\mu}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}}
$$
Let $a_1 = \mathbf{1}^T\Sigma^{-1}\mathbf{1}$ and $a_2 = \mathbf{1}^T\Sigma^{-1}r$ two scalars, we could rewrite the optimized solution for the portfolio weight vector $w$:
$$
w^* = \dfrac{1}{2\mu}(\Sigma^{-1})(r - \dfrac{\Sigma^{-1}}{2\mu}\Big( \dfrac{a_2}{a_1} - \dfrac{2\mu}{a_1} \Big)\mathbf{1})
$$

#### 1.1.3.2 $\varepsilon$-Constraint method

With this method we just keep one of the objectives and we restrict the rest of the objectives within user-specific values. Suppose that $x\in\mathbb{R}^n$,
$$
\begin{aligned}
	\min \quad & f_{\mu}(x)\\
		\textrm{s.t.} \quad & f_m(x) \leq \varepsilon_t, \quad t = 1,\dots,N, t\ne \mu\\ 
		& g_i(w) \geq 0, \quad i=1,\dots,m\\
        & h_j(w) = 0, \quad j=1,\dots,p\\
        & w_i^{(L)} \leq w_i\leq w_i^{(U)} \quad i = 1,\dots,k 
\end{aligned}

$$
For example, suppose that we have two o.f.. We could keep $f_2$ as the o.f., i.e. minimizing it, while treating $f_1$ as a constraint $f_1 \leq \varepsilon_1$.

<img src="C:\Users\Eric R\AppData\Roaming\Typora\typora-user-images\image-20210422173135546.png" alt="image-20210422173135546" style="zoom:80%;" />

The $\varepsilon$ vector has to be chosen carefully so that it is within the minimum or maximum values of the individual o.f..

## 1.2 Introduction to Portfolio Optimization

A portfolio is a collection of *N* assets $A_1, \dots, A_n$, wherein each security has an associated weight $w_1, \dots, w_n$. Is then possible to define a portfolio made up by these two components $P = \langle\mathcal{A}, W\rangle$, where $\mathcal{A}$ is the set of assets and $W$ is the vector of weights associated. 

Intuitively, a weight $w_i$ is the amount of wealth invested in an asset $A_i$ in a portfolio. Since we invest part of our initial investment in a standardized portfolio, the weights are proportional to it and so it holds that $\sum_{i=1}^N w_i = 1$.

The rate of return of the standardized portfolio is therefore a sum of random
variables, where the sum of weights is equals to one. 

Portfolio optimization plays a critical role in determining portfolio strategies for investors. What investors hope to achieve from portfolio optimization is to ***maximize portfolio returns and minimize portfolio risk***. Since return is compensated based on risk, investors have to balance the risk-return tradeoff for their investments. Therefore, there is no a single optimized portfolio that can satisfy all investors. An optimal portfolio is determined by an investor’s risk-return preference.

There are a few key concepts in portfolio optimization. First, *reward and risk are measured by expected return and variance of a portfolio*. **Expected return is calculated based on historical performance of an asset**, and **variance is a measure of the dispersion of returns**. Second, investors are exposed to two types of risk: *unsystematic risk* and *systematic risk*. Unsystematic risk is an asset’s intrinsic risk which can be diversified away by owning a large number of assets. These risks do not present enough information about the overall risk of the entire portfolio. Systematic risk, or the portfolio risk, is the risk generally associated with the market which cannot be eliminated. Third, the *covariance between different asset returns gives the variability or risk of a portfolio*. Therefore, a well-**diversified portfolio contains assets that have little or negative correlations**.

The key to achieving investors’ objectives is to provide an optimal portfolio strategy which shows investors how much to invest in each asset in a given portfolio. Therefore, the **decision variable** of portfolio optimization problems is the **asset weight** vector $w$. The expected return for each asset in the portfolio is expressed as $r = [r_1,r_2,\dots, r_n]^T$, where $p_i$ is the mean return for the asset *i*.

The portfolio expected return is the weighted average of the individual asset return $r_p = w^Tr = \sum_{i=1}^N w_ir_i$. Variance and covariance of an individual asset are characterized by a a covariance matrix $\Sigma$. The portfolio variance is then:
$$
\sigma_p^2 = w^T\Sigma w = \sum_{i=1}^N\sum_{j=1}^N w_iw_j\Sigma_{i,j}
$$

### 1.2.1 Portfolio optimization problem formulations

Modern portfolio theory assumes that for a given level of risk, a rational investor wants the maximal return, and for a given level of expected return, the investor wants the minimal risk. There are also extreme investors who only care about maximizing return (*disregard risk*) or minimizing risk (*disregard expected return*). There are generally five different formulations that serve investors of different investment objectives:
1. **Maximize expected return**, disregard risk
   $$
   \begin{aligned}
           \max \quad & r_p = r^Tw\\
           \textrm{s.t.} \quad & \mathbf{1}^Tw = 1
   \end{aligned}
   $$

2. **Minimize risk**, disregard expected return
   $$
   \begin{aligned}
           \min \quad & \sigma^2 = w^T\Sigma w\\
           \textrm{s.t.} \quad & \mathbf{1}^Tw = 1
   \end{aligned}
   $$

3. **Minimize risk for a given level of return $r^*$**
   $$
   \begin{aligned}
           \min \quad & \sigma_p^2 = w^T\Sigma w\\
           \textrm{s.t.} \quad & \mathbf{1}^Tw = 1\\
           & r^Tw = r^*
   \end{aligned}
   $$

4. **Maximize return for a given level of risk **$\sigma^{2*}$
   $$
   \begin{aligned}
           \max \quad & r_p = r^Tw\\
           \textrm{s.t.} \quad & \mathbf{1}^Tw = 1\\
           & w^T\Sigma w = \sigma^{2*}
   \end{aligned}
   $$

5. **Maximize return and minimize risk**
   $$
   \begin{aligned}
           \max\quad & r_p = r^Tw \,\,\text{and} \, \min \, \sigma_p^2 = w^T\Sigma w\\
           \textrm{s.t.} \quad & \mathbf{1}^Tw = 1
   \end{aligned}
   $$

We can note that model 3 and 4 are extensions of model 1 and 2 with fixed constraints. Model 3 is the model proposed by H. Markowitz, we can see that the formulation does not allow investors to simultaneously minimize risk and maximize expected return.

### 1.2.2 Portfolio multi-objective formulation

We will take as an example the fifth optimization problem, where we evaluate portfolio by maximizing return and minimizing risks. The specific formulation can be determined by recognizing that the two objectives minimizing portfolio's risk $\sigma_p^2 = w^T\Sigma w$ and maximizing portfolio's expected return $r_p = w^Tr$ are equivalent to minimizing the *negative* of the portfolio expected return and risk. Thus, this require a new formulation for the fifth problem:
$$
\begin{align}
        \min_{w} \quad & (f_1(w),f_2(w))=(-w^Tr, w^T\Sigma w)\\
        \textrm{s.t.} \quad & \mathbf{1}^Tw =1 \nonumber
\end{align}
$$
This multi-objective optimization can be solve using **scalarization** (see the weighted sum method), a standard technique for finding Pareto optimal points for any vector optimization problem by solving the ordinary scalar optimization. Assign two we weighting coefficients $\lambda_1, \lambda_2 >0$ for o.f. $f_1(w)$ and $f_2(w)$ respectively. By varying the two coefficients, one can obtain different Pareto optimal solutions of the vector optimization problem. Without loss of generality, one can take $\lambda_1=1$ and $\lambda_2 = \mu > 0$: 
$$
\begin{align}
        \min_{w} \quad & -w^Tr + \mu w^T\Sigma w\\
        \textrm{s.t.} \quad & \mathbf{1}^Tw =1 \nonumber
\end{align}
$$
The o.f. is indeed convex because $\Sigma$ is positive semi-definite. The $\mu$ coefficient represents how much an investor weights risk over expected return. One can consider $\mu$ as a risk aversion index that measures the risk tolerance of an investor. A smaller value of $\mu$ indicates that the investor is more risk-seeking, and a larger value of $\mu$ indicates that the investor is more risk-averse. All Pareto optimal portfolios can be obtained by varying $\mu$ except for two extreme cases where $\mu \rightarrow 0$ and $\mu \rightarrow \infty$. When $\mu \rightarrow0$, the o.f. is dominated by the expected return term, this represents the will of the investor to only maximize return and disregard risk (model 1). Symmetrically, when $\mu \rightarrow \infty$ the variance term will dominate the o.f. resulting in a portfolios that has minimal variance (model 2). So by varying $\mu$ one can generate various optimization models that serves investors of any kind of risk tolerance.

### 1.2.3 