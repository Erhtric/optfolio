"""
THIS FILE CONTAINS THE METHODS FOR THE SIMPLEX ALGORITHM EXECUTION.
"""
import numpy as np
class Simplex:
    """
    This class includes the Simplex algorithm and the method associated. As it is, it can
    be used to solve any linear optimization problem. In this specific project, it will be
    used to solve a portfolio o/ptimization problem, even though there are faster method to do so.
    """

    def __init__(self, c, A, b):
        """
        Initializes a Simplex session in the standard form
            max c.T @ X s.t. A @ X = b
        where:
            - c is the vector of coefficients for the objective function
            - A is the matrix of coefficients for the constraint equations
                + coefficients of the slack variables
            - b is the vector of constants on the RHS of the constraint equations
        """
        # Array of coefficients of the objective function
        self.of_params = c
        # Matrix of coefficients of the constraints
        self.bounds_params = A
        # Array of constants
        self.constants = b

        # Array of solutions
        # since the tableau is ordered from left to right we need only the number of original variables in the problem
        self.n_vars = np.count_nonzero(self.of_params)
        self.solutions = np.zeros(self.n_vars)
        self.slack = np.zeros(self.constants.shape[0])
        # value of the objective function
        self.objective = 0

        # USED FOR INTERNAL USE: it is the internal tableau built with the relative method
        self.tableau = []
        self.iteration = 0
        # Flag for setting the maximization/problem
        self.maximization = True

    def create_tableau(self):
        """
        Create the tableau. From left to right, the column identify a different
        variable or constant; indeed we will have all the non-basic variables,
        then the slack variables.

        tableau\n
        = [[X S] b
            -c    0]
        where z is the coefficient relative to objective function
        """
        Ab = np.array([np.concatenate((p, b)) for p, b in zip(self.bounds_params, self.constants.T)], dtype=np.float64)
        z = np.array(np.concatenate((-self.of_params, [0])).reshape((Ab.shape[1],1)), dtype=np.float64)        # the objective function should be correctly negated
        self.tableau = np.concatenate((Ab, z.T), axis = 0)

    def __is_optimal(self):
        """A solution is optimal if in every term in the objective function is non-negative.
        This method perform a simple check on the last row of the tableau if there an element
        which is < 0.
        """
        c = self.tableau[-1, :-1]
        return all(val >= 0 for val in c)

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

    def __sub_pivoting_1(self, row_idx, col_idx):
        """Part 1 of the pivot-changing part: the goal of this
        method is to set the pivot to 1 by multiplying the corresponding row by a certain factor"""
        pivot = self.tableau[row_idx, col_idx]
        self.tableau[row_idx, :] = self.tableau[row_idx, :] / pivot
        print(f'Dividing by {pivot}')

    def __sub_pivoting_2(self, row_idx, col_idx):
        """Part 2 of the pivot-changing part: the goal of this
        method is to set the terms in the colum, apart from the pivot to 0"""
        for i in range(self.tableau.shape[0]):
            row = self.tableau[i, :]
            # To set the term in the pivot column to zero we have to subtract the
            # value from the entire row.
            print(f'Subtracting by {row[col_idx]}')
            if i != row_idx and row[col_idx] != 0:
                # The operation is: R_i = R_i - mul * R_p
                pivot_row = self.tableau[row_idx, :]
                multiplier = row[col_idx]
                self.tableau[i, :] = row - multiplier * pivot_row

    def apply_pivoting(self):
        """This method apply the pivoting step to the tablueau by
        finding an appropriate pivot to then change the accordingly the values.
        This method modifies the tableau values.
        """
        row, col = self.get_pivot_position()
        print(f'Pivoting on row {row} and column {col}')

        self.__sub_pivoting_1(row, col)
        self.__sub_pivoting_2(row, col)

    # def __repr__(self):
    #     print(f'Tableau at iteration {self.iteration}:\n')
    #     print(self.tableau)

    def extract_solution(self):
        """This method guess the solution from the tableau which has to be optimal.
        A solution is composed by the non-basic variables' coefficient that appear in the first and the
        last optimal tableau and its relative constant values.
        """
        for col in range(self.tableau.shape[1]):
            if self.__is_basic(col):
                idx = np.argmax(self.tableau[:, col])
                print(f'idx: {idx} col: {col}')
                if col < self.n_vars:
                    # this must be included in the solutions list
                    self.solutions[col] = self.tableau[idx, -1]
                else:
                    # this must be included in the slack list
                    self.slack[col] = self.tableau[idx, -1]
            else:
                self.solutions[col] = 0
        self.objective = self.tableau[-1, -1]

    def simplex(self):
        """Main method of the class. It needs to be called in order to get
        an array of solutions. It iteratively search for a solution by applying a pivoting operation
        to the tableau until the current set of solutions is optimal.
        """
        self.create_tableau()
        print(self.tableau)

        while not self.__is_optimal():
            self.apply_pivoting()
            print(self.tableau)
            self.iteration += 1

        self.extract_solution()
        return self.solutions