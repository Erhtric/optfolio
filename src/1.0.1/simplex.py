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
        """
        self.c = c
        self.A = A
        self.b = b

    def initialize_matrix(self):
        """
        Form the structured matrix containing all the coefficients, the slack variables,
        and the values associated
        """
        pass

    def create_tableau(self):
        """
        Create the tableau given the actual values of c, A and b

        tableau = [ A b
                           c 0 ]
        """
        Ab = np.array([np.concatenate([eq, bb]) for eq, bb in zip(self.A, self.b.T)])
        obj = np.concatenate((self.c, [0])).reshape((Ab.shape[1],1))
        return np.concatenate((Ab, obj.T), axis=0)

    def is_not_optimal(self, tableau):
        """Checks if the current tableau provides a not ideal solution,
        namely the fact if there exists variables that needs to be lowered

        Args:
            tableau: bi-dimensional array representing a tableau
        """
        obj = tableau[-1]
        return np.any([obj[i] > 0 for i in range(len(obj))])

    def compute_pivot_position(self):
        pass

    def apply_step(self):
        pass

    def get_solution(self):
        pass

    def simplex(self):
        tableau = self.create_tableau()

        while is_not_optimal(tableau):
            position = compute_pivot_position(tableau)
            tableau = apply_step(tableau, position)

        solution = get_solution(tableau)