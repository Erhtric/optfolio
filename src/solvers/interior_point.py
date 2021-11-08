"""
THIS FILE CONTAINS THE METHODS FOR THE INTERIOR POINT ALGORITHM EXECUTION for the PORTFOLIO OPTIMIZATION.
"""
import numpy as np
import pandas as pd

class IntPoint:
    """This class contains an implementation of a modified version of the Mehrotra predictor-corrector method for linear programming.
    Thanks to some modifications the method is now suited for convex quadratic programming problems.
    """

    def __init__(self
                , S: np.array
                , A: np.array
                , b: np.array
                , c
                , max_iteration=1000
                , epsilon = 1.0e-5
                , max=True
                , verbose=False) -> None:
        """
        Initializes an Interior Point session in the standard form for solving a quadratic program
            min x^T S x + x^T c s.t. A @ x >= b
        where:
            - S is a symmetric positive semi-definite matrix
            - A is the matrix of coefficients for the constraint equations
            - b is the vector of constants on the RHS of the constraint equations
        This code follows the algorithm presented in Nocedal & Wright (2006)[Numerical Optimization]
        """
        self.S = S
        self.A = A
        self.b = b.reshape((b.shape[0], 1))
        self.c = c

        self.n_vars = self.A.shape[1]
        self.n_eq   = self.A.shape[0]

        self.iteration      = 0
        self.verbose        = verbose
        self.max_iteration  = max_iteration
        self.epsilon        = epsilon          # for the tolerance

        # Initialization variables: user-defined starting point
        # TODO generate it randomically and check if respect the constraints
        self.x_0  = np.ones(self.n_vars)
        self.y_0  = 2 * np.ones(self.n_vars)        # slack
        self.lm_0 = 3 * np.ones(self.n_vars)        # langrangian multiplier

    def corrector_step(self, x_k, y_k, lm_k, Gamma_aff, Lambda_aff, sigma):
        """It computes the affine scaling step (w_aff, y_aff, lm_aff) from the point (x_k, y_k, lm_k)
        by solving the system built with the perturbed KKT conditions for the given convex quadratic program.

        The linear system is of the form:
        [G  0 -A.T [d_x   [      -rd
         A -I  0    d_y  =       -rp
         0  LA GA ] d_lm]  -LA @ GA @ e + sigma @ mu @ e]
        Obtained by fixing mu and applying the Newton's method to KKT conditions.
        """
        Gamma = np.diag(y_k)
        Lambda = np.diag(lm_k)
        print(x_k, y_k, lm_k)
        rd = self.S @ x_k - self.A.T @ lm_k + self.c
        rp = self.A @ x_k - y_k - self.b
        mu = (y_k @ lm_k) / self.n_eq         # the complementarity measure
        e = np.zeros(self.n_eq)

        LHS = np.block([
            [self.S, np.zeros((self.n_vars, self.n_eq)), -self.A.T],
            [self.A, -np.eye(self.n_eq), np.zeros((self.n_eq, self.n_eq))],
            [np.zeros((self.n_eq, self.n_vars)), Lambda, Gamma]
        ])

        print(LHS.shape)
        RHS = np.block([
            -rd,
            -rp,
            -Lambda @ Gamma @ e - Lambda_aff @ Gamma_aff @e + sigma * mu * e
        ])
        print(RHS)

    def solve(self):
        dx_aff, dy_aff, dlm_aff = self.corrector_step()


