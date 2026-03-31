#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a class of linear time invariant
# systems
#

# jax
import jax.numpy as jnp

# scipy are solver
from scipy.linalg import solve_continuous_are

# nonlinear system class
from helpers.nonlinear_system import NonlinearAffineSystem

class LTI(NonlinearAffineSystem):

    def __init__(
            self,
            A: jnp.ndarray = None,
            B: jnp.ndarray = None,
            C: jnp.ndarray = None,
            D: jnp.ndarray = None,
            initial_state: jnp.ndarray = None,
            ):

        if A is None:
            # A = - jnp.array([[1., 2.], [3., 4.]])
            A = jnp.array([[0,1],[-1,0]]) + 0.1 * jnp.eye(2)
            # A = 1/2 * jnp.array([[1, 1], [-1, -1]])
        if B is None:
            B = jnp.array([[0.], [1.]])
        if C is None:
            C = jnp.array([[1., 0.]])
            # C = B.T
        if D is None:
            D = jnp.array([0.])

        self.A_matrix = A
        self.B_matrix = B
        self.C_matrix = C
        self.D_matrix = D

        if initial_state is None:
            initial_state = jnp.ones((A.shape[1],))

        super().__init__(
            f=lambda z: self.A_matrix @ z,
            g=lambda z: self.B_matrix,
            h=lambda z: self.C_matrix @ z,
            k=lambda z: self.D_matrix,
            initial_state=initial_state,
            ncontrol=B.shape[1]
            )

        return

class LTI_withOptimalControl(LTI):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        assert jnp.allclose(self.D_matrix, 0.0)

        self.h_orig = lambda z: self.C_matrix @ z # position observation

        # get value function of optimal control problem via a riccati equation
        # solve control ARE
        cARE_solution = solve_continuous_are(a=self.A_matrix, b=self.B_matrix, q=self.C_matrix.T @ self.C_matrix, r=jnp.eye(self.ncontrol)) # scipy method
        self.cARE_solution = jnp.array(cARE_solution) # take jax array

        self.value_function = lambda z: 1/2 * z.T @ self.cARE_solution @ z
        self.gradient_value_function = lambda z: self.cARE_solution @ z

        # overwrite the output of the system to make it dissipative
        # with respect to the supply rate
        #
        # s(u,yhat) = 1/2 * yhat^T yhat + yhat^T u
        #
        # where yhat = B(z)^T \nabla V(z) is the feedback law
        self.h = lambda z: self.B(z).T @ self.gradient_value_function(z)

        # set hamiltonian and eta parameters for consistency with other parts of the codebase
        self.hamiltonian = self.value_function
        self.eta = self.gradient_value_function

        return

class PI_Controller(LTI):

    def __init__(
            self,
            k_I: float = 1.,
            k_P: float = 1.,
            ):

        self.k_I = k_I
        self.k_P = k_P

        I = jnp.eye(1)

        super().__init__(
            A = 0*I,
            B = I,
            C = self.k_I * I,
            D = self.k_P * I,
            )

        self.hamiltonian = lambda z: 1/2 * self.k_I * z.T @ z
        self.eta = lambda z: self.k_I * z

        return

if __name__ == "__main__":
    
    pass
