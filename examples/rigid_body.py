#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the spinning rigid body example
#
#

import jax
import jax.numpy as jnp

# energy based model
from helpers.energy_based_model import EnergyBasedModel

# nonlinear system
from helpers.nonlinear_system import NonlinearAffineSystem_NoFeedthrough

class RigidBody(EnergyBasedModel, NonlinearAffineSystem_NoFeedthrough):
    
    def __init__(
            self,
            I: jnp.ndarray = jnp.array([1.0, 2.0, 3.0]),
            b: jnp.ndarray = jnp.array([1.0, 1.0, 1.0]),
            ):
        
        # assemble Q matrix
        self.Q_matrix = jnp.diag(1/I)
        self.Qinv_matrix = jnp.diag(I)
        
        # assemble B matrix
        self.B_matrix = b.reshape((3, 1))
        
        # call super
        super().__init__()
        
        # set dimensions
        self.dims = (0, 3, 0)
        
        # set initial condition
        self.initial_condition = jnp.array([0., 0.5, 1.])
        
        # for discrete gradient method
        self.eta = lambda z: self.Q_matrix @ z
        self.ham_eta = lambda z: (self.hamiltonian(jnp.zeros_like(z),z), self.eta(z))
        self.r = lambda v: - self.J(jnp.zeros_like(v), v, jnp.zeros_like(v)) # map f(z) = -r(eta(z))
        NonlinearAffineSystem_NoFeedthrough.__init__(
            self,
            f = lambda z: self.J(jnp.zeros_like(z),self.Q_matrix @ z, jnp.zeros_like(z)),
            g = lambda z: self.B_matrix,
            h = lambda z: self.B_matrix.T @ self.Q_matrix @ z,
            initial_state = self.initial_condition,
            ncontrol = 1,
        )
        
        
    def J(self, dt_z1, h2, z3):
        
        p_x, p_y, p_z = self.Qinv_matrix.dot(h2) # reconstruct state
        J = jnp.zeros((3,3))
    
        J = J.at[0,:].set([0, -p_z, p_y])
        J = J.at[1,:].set([p_z, 0, -p_x])
        J = J.at[2,:].set([-p_y, p_x, 0])
        
        return J @ h2
    
    def R(self, dt_z1, h2, z3):
        return 0 * h2
    
    def hamiltonian(self, z1, z2):
        # z1 not needed
        return 1/2 * z2.T @ self.Q_matrix @ z2
    
    def B(self, u):
        return self.B_matrix @ u

    def default_control(self, t):
        return jnp.sin(2*t).reshape((t.shape[0], 1))
    
    def get_manufactured_solution(self):
        
        def control_manufactured_solution(t):
            # t is scalar, return shape (1,) for consistency
            return jnp.sin(2*t).reshape((1,))
        
        def manufactured_solution(t):
            return jnp.hstack((jnp.sin(t), jnp.cos(t) - 0.5, jnp.sin(t)*jnp.cos(t) + 1))
        
        def dt_manufactured_solution(t):
            return jnp.hstack((jnp.cos(t), -jnp.sin(t), jnp.cos(2*t)))
        
        def g_manufactured_solution(t):
            # t is scalar
            u = control_manufactured_solution(t)
            z = manufactured_solution(t)
            dt_z = dt_manufactured_solution(t)
            _ = jnp.zeros((0,))
            rhs = self.J(_, self.nabla_2_ham(_, z), _) + self.B(u)
            return dt_z - rhs
            
        # z0 should be initialized with scalar input
        z0 = manufactured_solution(0.0)
        
        return (
            z0,
            jax.vmap(manufactured_solution, in_axes=0),
            jax.vmap(control_manufactured_solution, in_axes=0),
            jax.vmap(g_manufactured_solution, in_axes=0),
        )

        

if __name__ == "__main__":
    
    pass