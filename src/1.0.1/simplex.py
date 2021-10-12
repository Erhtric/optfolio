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

    feasible = None
    variables = None
    obj_f = None

    def __init__(self, c, A, b):
        """
        Initializes a Simplex session in the standard form
            max c.T @ X s.t. A @ X = b
        """
        self.c = 0
        self.A = 0
        self.b = 0

    def initialize_matrix(self):
        """
        Form the structured matrix containing all the coefficients, the slack variables,
        and the values associated
        """
        pass

    def create_tableau(self):
        """
        Create the tableau given the actual values of c, A and b
        """
        xb = np.array([np.concatenate([eq, bb]) for eq, bb in zip(A, b.T)])
        z = np.concatenate((c, [0])).reshape((xb.shape[1],1))
        return np.concatenate((xb, z.T), axis=0)
        
    def is_optimal(self, tableau):
        pass

    def simplex(self):
        tableau = self.create_tableau()

        while is_optimal(tableau):
            pass