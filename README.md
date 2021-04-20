# optfolio
[TOC]

# 1. Background

## 1.1 Modern Portfolio Theory 

The ***Modern Portfolio Theory*** (**MPT**) refers to  an investment theory that allows investors to assemble an asset portfolio that maximizes expected return for a given level of risk. The theory assumes that investors are risk-averse (but this is a choice since there exists portfolios that maximizes risks and return rates).

According to the MPT

## 1.2 Pearson's Correlation



## 1.3 Mean-Variance Analysis

**Mean-variance analysis** is a component of the ***<u>Modern Portfolio Theory</u>*** (**MPT**) is a *technique that investors use to make decisions about financial  instruments to invest in, based on the amount of risk that they are  willing to accept (**risk tolerance**)*. Investors use mean-variance analysis to make investment decisions. Investors weigh how much risk they are willing to take on in exchange for different levels of reward.  Mean-variance analysis allows investors to find the biggest reward at a given level of risk or the least risk at a given level of return.

Ideally, investors expect to earn higher returns when they invest in riskier assets, in fact when measuring the level of risk, investors consider the **potential variance**, namely the *volatility of returns produced by an asset*, against the expected returns of that asset. Basically what mean-variance analysis does it to look at the *average variance in the expected return from an investment*.

![Mean-Variance Analysis](https://cdn.corporatefinanceinstitute.com/assets/mean-variance-analysis.png)

When choosing a financial asset to invest in, investors prefer the asset with *lower variance* when given choosing between two otherwise identical investments. An investor can achieve diversification by investing in securities with varied variances and expected returns. Proper diversification creates a portfolio where a loss in one security is counter-balanced by a gain in another.

Mean-variance analysis is made up by two main components, as follows:

- the *variance*, which is the ***measure of how much distant or spread are the numbers in a data set from the mean***. When speaking of an investment portfolio, variance can show how the returns of a security are spread out during a given period.
- the *expected return*, which is the estimated return that a security is expected to produce. In the real word, the expected rate of return is based on historical data so it is no guaranteed.

Given that the variance of two assets is somewhat equal, the investor should choose the one which offers the highest expected return. 

#### Example.

Assume a portfolio comprised of the following two stocks:

1. A:  $100,000 with an expected return of 5%.
2. B:  $300,000 with an expected return of 10%.

The total value of the portfolio is the sum of the components, then $400,000​, and the weight of each stock is computed as follows:

1. A: $\dfrac{\$100000}{\$400000} = 0.25 = 25\%$
2. B: $\dfrac{\$300000}{\$400000} = 0.75 = 75\%$

Then, the expected rate of return, namely the weight of the asset in the portfolio multiplied by the expected return, is obtained as follows, defined $M$ as the set of the assets present in the portfolio:

$$\begin{align*}
R_{tot} &= \sum_{i \subseteq M} w_i R_i \\&= (0.25 \times 0.05) + (0.75 \times 0.1)\\
  &= 0.0125 + 0.075\\
  &= 0.0875 = 8.75\%
\end{align*}$$

Portfolio variance is more complicated to calculate because it is not a simple weighted average of the investments' variances. We firstly need to calculate the correlation between the two assets, you could use ***Pearson's Correlation***, which in this case is equal to $0.65$. Then, you can compute form the historical data standard deviation for the asset A is $\sigma_A = 0.07$ while for asset B is $\sigma_B = 0.14$. From this is possible to compute the *portfolio variance*:

$$\begin{align*}
\sigma^2_{portfolio} &= \sum_{i \subseteq M} w_i^2 \sigma_i ^2 + |M|\sum_{i,j \subseteq M, i\neq j} w_i\sigma_i\rho_{i, j} \quad\text{this part needs rework}\\
  &=(0.25^2 \times 0.07^2) + (0.75^2 \times 0.14^2) \\&\quad+ (2 \times 0.25 \times 0.75 \times 0.07 \times 0.14 \times 0.65) \\
  &= 0.0137
\end{align*}$$

Whereas the *portfolio standard deviation* is simply $\sigma = \sqrt{\sigma^2_{portfolio}} = 0.1171$.
