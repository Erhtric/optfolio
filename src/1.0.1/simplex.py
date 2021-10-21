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

    def __is_optimal(self):
        """A solution is optimal if in every term in the objective function is non-negative.
        This method perform a simple check on the last row of the tableau if there an element
        which is < 0, and if it this is true it returns false.
        """
        z = self.tableau[-1, :-1]
        return all(val >= 0 for val in z)

    def __is_basic(self, idx):
        """Checks if the column at index idx is a unit-column, then the variable associated
        is a basic variable. If it is not the case the variable it is a non-basic one.
        """
        return np.sum(self.tableau[:, idx]) == 1

    def __compute_pivot_col_position(self):
        """This method simply computes the column index for the pivot in the tableau.
        If a solution is non optimal, then one or more terms in the last row of the tableau
        are negative, this is done by taking the minimum value among those values.
        """
        z = self.tableau[-1]
        return np.argmin(z, axis=0) if np.min(z) < 0 else None

    def __compute_pivot_row_position(self):
        """This method computes the row index for the pivot in the tableau.
        If a solution is non optimal, and given the pivoting column the index for the
        row will be the minimum value obtained as the result from dividing the corresponding
        constant by the target element indicized by the row pivoting index.
        """
        col_idx = self.__compute_pivot_col_position()
        A = self.tableau[:-1, :]
        temp = []
        for row in A:
            target = row[col_idx]
            b = row[-1]
            # this last operation is crucial: since we have to divide by the elements
            # in the coefficient matrix it is important to identify the elements which are
            # equal to zero, in addition to the one which are not interesting for the algorithm
            # which are the negative ones (tagged as infinite)
            temp.append(np.inf if target <= 0 else b / target)

        return np.argmin(temp)

    def get_pivot_position(self):
        """Computes the pivot position for the current tableau.
        """
        return self.__compute_pivot_row_position(), self.__compute_pivot_col_position()

    def sub_pivoting_1(self, row_idx, col_idx):
        """Part 1 of the pivot-changing part: the goal of this
        method is to set the pivot to 1 by multiplying the corresponding row by a certain factor"""
        pivot = self.tableau[row_idx, col_idx]
        self.tableau[row_idx, :] = self.tableau[row_idx, :] / pivot

    def sub_pivoting_2(self, row_idx, col_idx):
        """Part 2 of the pivot-changing part: the goal of this
        method is to set the terms in the colum, apart from the pivot to 0"""
        for i in range(self.tableau.shape[0]):
            # The row with the pivot must be avoided
            if i != row_idx:
                row = self.tableau[i, :]
                multiplier = self.tableau[row_idx, :] * self.tableau[i, col_idx]
                row = row - multiplier

    def apply_pivoting(self):
        """[summary]
        """
        row, col = self.get_pivot_position()

        self.sub_pivoting_1(row, col)
        self.sub_pivoting_2(row, col)
