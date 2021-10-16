import numpy as np
class Simplex:
    """
    Simplex algorithm steps:
        1. standard form, all the constraints are equations and all the variables are nonnegative.
        2. slack variables
        3. creating the tableau,
        4. pivot variables
        5. create a new tableau,
        6. checking for optimality,
        7. identify optima values
    """

    def __init__(self, c, A, b):
        """
        Initializes a Simplex session in the standard form
            max c.T @ X s.t. A @ X = b
        where we can identifiy the relative objective function to be maximized and the constraints associated
        """
        self.c = c
        self.A = A
        self.b = b

    def create_tableau(self):
        """
        Create the tableau given the actual values of c, A and b

        tableau = [ A b
                           c 0 ]
        """
        A_b = np.array([np.concatenate([eq, bb]) for eq, bb in zip(self.A, self.b.T)])
        obj = np.concatenate((self.c, [0])).reshape((A_b.shape[1],1))
        return np.concatenate((A_b, obj.T), axis=0)

    def next_iteration(self, tableau):
        """
        Checks if the current tableau provides a not ideal solution,
        namely the fact if there exists variables that needs to be lowered

        Args:
            tableau: bi-dimensional array representing a tableau
        """
        obj = tableau[-1]
        # np.any(val > 0 for val in obj[:-1])
        #TODO
        return np.any([obj[i] >= 0 for i in range(len(obj)-1)])

    def is_pivoting_right(self, tableau):
        """
        Checks if the tableau's furthest right column has negative values, thus it needs pivoting
        """
        column = tableau[:-1, -1]        # exclude the last value
        return np.min(column) <= 0

    def is_pivoting_bottom(self, tableau):
        """
        Checks if the tableau's bottom row has negative values, thus it needs pivoting
        """
        row = tableau[-1, :-1]          # exclude the last value
        return np.min(row) <= 0

    def compute_neg_position_right(self, tableau):
        """
        This function determines where a pivot element is located in the furthest right column
        """
        col = tableau[:-1, -1]
        res = np.argmin(col, axis=1) if np.min(col) <= 0 else None
        return res

    def compute_neg_position_bottom(self, tableau):
        """
        This function determines where a pivot element is located in the bottom row, obj
        """
        row = tableau[-1, :-1]
        res = np.argmin(row, axis=1) if np.min(row) <= 0 else None
        return res

    def compute_pivot_position_right(self, tableau):
        #TODO
        acc = []
        neg = self.compute_neg_position_right(tableau)
        row = tableau[neg, :-1]
        idx_min = np.argmin(row)
        col = tableau[:-1, idx_min]
        b_col = tableau[:-1, -1]

        for el, b in zip(col, b_col):
            if b / el > 0:
                acc += b / el
            else:
                acc += np.inf

        idx = acc.index(np.min(acc))
        return idx, idx_min

    def compute_pivot_position(self, tableau):
        if self.is_pivoting_bottom(tableau):
            acc = []
            neg = self.compute_neg_position_bottom(tableau)
            for el, b in zip():
                if b / el > 0:
                    acc += b / el
                else:
                    acc += np.inf
            idx = acc.index(np.min(acc))
            return idx, neg

    def apply_step(self, tableau, position):
        pass

    def get_solution(self, tableau):
        pass

    def simplex(self):
        tableau = self.create_tableau()

        while next(tableau):
            position = compute_pivot_position(tableau)
            tableau = apply_step(tableau, position)

        solution = get_solution(tableau)