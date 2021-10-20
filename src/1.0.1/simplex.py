import numpy as np
class Simplex:
    """
    This class includes the Simplex algorithm and the method associated. As it is, it can
    be used to solve any linear optimization problem. In this specific project, it will be
    used to solve a portfolio optimization problem, even though there are faster method to do so.
    """

    def __init__(self, c, A, b):
        """
        Initializes a Simplex session in the standard form
            max c.T @ X s.t. A @ X = b
        where:
            - c is the vector of coefficients for the objective function
            - A is the matrix of coefficients for the constraint equations + coefficients of the slack variables
            - b is the vector of constants on the RHS of the constraint equations
        """
        # Array of coefficients of the objective function
        self.of_params = c
        # Matrix of coefficients of the constraints
        self.bounds_params = A
        # Array of constants
        self.constants = b
        # Flag for setting the maximization/problem
        self.maximization = True

        # self.slack_vars = np.zeros(self.constants.shape[0])

        # Array of solutions
        self.solutions = np.zeros(self.of_params.shape[0])
        # USED FOR INTERNAL USE: it is the internal tableau built with the relative method
        self.tableau = []

    def create_tableau(self):
        """
        Create the tableau. From left to right, the column identify a different variable or constant; indeed
        we will have all the non-basic variables, then the slack variables.

        tableau\n
        = [[X S] b
            -c    0]
        where z is the coefficient relative to objective function
        """
        Ab = np.array([np.concatenate((p, b)) for p, b in zip(self.bounds_params, self.constants.T)])
        z = np.concatenate((-self.of_params, [0])).reshape((Ab.shape[1],1))        # the objective function should be correctly negated
        self.tableau = np.concatenate((Ab, z.T), axis = 0)

    def is_optimal(self):
        """A solution is optimal if in every term in the objective function is non-negative.
        This method perform a simple check on the last row of the tableau if there an element
        which is < 0, and if it this is true it returns false.
        """
        z = self.tableau[-1, :-1]
        return all(val >= 0 for val in z)

    def get_pivot_col_position(self):
        """This method simply computes the column index for the pivot in the tableau.
        If a solution is non optimal, then one or more terms in the last row of the tableau
        are negative, this is done by taking the minimum value among those values.
        """
        if not self.is_optimal():
            return None
        z = self.tableau[-1]
        return np.argmin(z, axis=1) if np.min(z) < 0 else None

    def get_pivot_row_position(self):
        """This method computes the row index for the pivot in the tableau.
        If a solution is non optimal
        """
        col_idx = self.get_pivot_col_position()
        b = self.tableau[:-1, -1]
        b = b / self.tableau[:-1, col_idx]
        return np.argmin(b, axis=1) if np.min(b) >= 0 else None
























































    # def is_pivoting_right(self, tableau):
    #     """
    #     Checks if the tableau's furthest right column has negative values, thus it needs pivoting
    #     """
    #     column = tableau[:-1, -1]        # exclude the last value
    #     return np.min(column) <= 0

    # def is_pivoting_bottom(self, tableau):
    #     """
    #     Checks if the tableau's bottom row has negative values, thus it needs pivoting
    #     """
    #     row = tableau[-1, :-1]          # exclude the last value
    #     return np.min(row) <= 0

    # def compute_neg_position_right(self, tableau):
    #     """
    #     This function determines where a pivot element is located in the furthest right column
    #     """
    #     col = tableau[:-1, -1]
    #     res = np.argmin(col, axis=1) if np.min(col) <= 0 else None
    #     return res

    # def compute_neg_position_bottom(self, tableau):
    #     """
    #     This function determines where a pivot element is located in the bottom row, obj
    #     """
    #     row = tableau[-1, :-1]
    #     res = np.argmin(row, axis=1) if np.min(row) <= 0 else None
    #     return res

    # def compute_pivot_position_right(self, tableau):
    #     #TODO
    #     acc = []
    #     neg = self.compute_neg_position_right(tableau)
    #     row = tableau[neg, :-1]
    #     idx_min = np.argmin(row)
    #     col = tableau[:-1, idx_min]
    #     b_col = tableau[:-1, -1]

    #     for el, b in zip(col, b_col):
    #         if b / el > 0:
    #             acc += b / el
    #         else:
    #             acc += np.inf

    #     idx = acc.index(np.min(acc))
    #     return idx, idx_min

    # def compute_pivot_position(self, tableau):
    #     if self.is_pivoting_bottom(tableau):
    #         acc = []
    #         neg = self.compute_neg_position_bottom(tableau)
    #         for el, b in zip():
    #             if b / el > 0:
    #                 acc += b / el
    #             else:
    #                 acc += np.inf
    #         idx = acc.index(np.min(acc))
    #         return idx, neg

    # def apply_step(self, tableau, position):
    #     pass

    # def get_solution(self, tableau):
    #     pass

    # def simplex(self):
    #     tableau = self.create_tableau()

    #     while next(tableau):
    #         position = compute_pivot_position(tableau)
    #         tableau = apply_step(tableau, position)

    #     solution = get_solution(tableau)
