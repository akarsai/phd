#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a nonlinear pendulum as a port-hamiltonian system
#
#

# jax
import jax.numpy as jnp

# nonlinear system class
from helpers.nonlinear_system import NonlinearAffineSystem_NoFeedthrough

class Pendulum(NonlinearAffineSystem_NoFeedthrough):

    def __init__(
            self,
            initial_state = None,
            friction_coefficient = 0.2,
            gravitation = 9.81,
            ):
        """
        models a nonlinear pendulum with friction and velocity control

        the energies of the system is E = E_kin + E_pot with

        E_kin = 1/2 * m * l^2 * theta_dot^2
        E_pot = m * g * l * (1 - cos(theta)).

        we take m = l = 1 in the following.

        the dynamics read as

        d^2/dt^2 theta = - g * sin(theta) - friction_coefficient * d/dt theta + u

        after order reduction, we arrive at

        d/dt theta = omega
        d/dt omega = - g * sin(theta) - friction_coefficient * omega + u

        setting z = (theta, omega), we arrive at a port-hamiltonian representation

        d/dt z = (J - R) eta(z) + B u

        Args:
            initial_state: (optional) initial state of the pendulum
            friction_coefficient: (optional) friction coefficient
        """

        self.gravitation = gravitation
        self.friction_coefficient = friction_coefficient

        self.J_matrix = jnp.array([[0, 1], [-1, 0]])
        self.R_matrix = jnp.array([[0, 0], [0, self.friction_coefficient]])
        self.B_matrix = jnp.array([[0],[1]])

        self.hamiltonian = lambda z: self.gravitation * (1-jnp.cos(z[0])) + 1/2 * z[1]**2
        self.eta = lambda z: jnp.array([self.gravitation * jnp.sin(z[0]), z[1]])
        self.ham_eta = lambda z: (self.hamiltonian(z), self.eta(z))
        self.r = lambda v: - (self.J_matrix - self.R_matrix) @ v # map f(z) = -r(eta(z))

        if initial_state is None:
            initial_state = jnp.array([1/4 * jnp.pi, -1.0])

        super().__init__(
            f=lambda z: (self.J_matrix - self.R_matrix) @ self.eta(z),
            g=lambda z: self.B_matrix,
            h=lambda z: self.B_matrix.T @ self.eta(z),
            initial_state=initial_state, ncontrol=1
            )

        return
    
if __name__ == "__main__":
    
    pass