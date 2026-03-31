#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the toda lattice example
#
#

import jax
import jax.numpy as jnp

# energy based model
from helpers.energy_based_model import EnergyBasedModel_LinearJR

# nonlinear system
from helpers.nonlinear_system import NonlinearAffineSystem_NoFeedthrough

class Toda(EnergyBasedModel_LinearJR, NonlinearAffineSystem_NoFeedthrough):
    
    def __init__(
            self,
            number_of_particles: int = 5,
            gamma: float = 0.1,
            ):
        
        self.number_of_particles = number_of_particles
        self.gamma = gamma
        
        # assemble J matrix
        Z = jnp.zeros((self.number_of_particles,self.number_of_particles))
        I = jnp.eye(self.number_of_particles)
        J1 = jnp.hstack((Z,I))
        J2 = jnp.hstack((-I,Z))
        J_matrix = jnp.vstack((J1,J2))
        
        # assemble R matrix
        R1 = jnp.hstack((Z,Z))
        R2 = jnp.hstack((Z,gamma * I))
        R_matrix = jnp.vstack((R1,R2))
        
        # call super (this is EnergyBasedModel_LinearJR)
        super().__init__(J_matrix=J_matrix, R_matrix=R_matrix)
        
        # assemble B matrix
        b = jnp.zeros((self.number_of_particles,1)).at[0,:].set(1)
        self.B_matrix = jnp.vstack((0*b,b))
        
        # set dimensions
        self.dims = (0, 2*self.number_of_particles, 0)
        
        # set initial condition
        self.initial_condition = jnp.hstack((jnp.arange(self.number_of_particles), jnp.zeros((self.number_of_particles,))))
        
        # for discrete gradient method
        self.eta = lambda z: self.nabla_2_ham(jnp.zeros_like(z), z)
        self.r = lambda v: - (self.J_matrix - self.R_matrix) @ v
        self.ham_eta = lambda z: (self.hamiltonian(jnp.zeros_like(z), z), self.eta(z))
        NonlinearAffineSystem_NoFeedthrough.__init__(
            self,
            f = lambda z: (self.J_matrix - self.R_matrix) @ self.eta(z),
            g = lambda z: self.B_matrix,
            h = lambda z: self.B_matrix.T @ self.eta(z),
            initial_state = self.initial_condition,
            ncontrol = 1
            )
        
        
        
    def hamiltonian(self, z1, z2):
        # z1 not needed, fixed typo from chaturantabut, beattie, gugercin paper
        
        # get q and p
        q = z2[:self.number_of_particles]
        p = z2[self.number_of_particles:]
    
        # calculate hamiltonian
        return 1/2 * p.T @ p \
                + jnp.sum(jnp.exp(q[:-1] - q[1:])) \
                + jnp.exp(q[-1] - q[0]) - self.number_of_particles
    
    def B(self, u):
        return self.B_matrix @ u
    
    def default_control(self, t):
        return jnp.sin(2*t).reshape((t.shape[0],1))
    
    def get_manufactured_solution(self):
        
        def control_manufactured_solution(t):
            # t is scalar, return shape (1,) for consistency
            return jnp.sin(2*t).reshape((1,))
        
        def q_manufactured(t):
            # t should be scalar here
            return jnp.sin(t) * jnp.ones((self.number_of_particles,))
    
        def dt_q_manufactured(t):
            return jnp.cos(t) * jnp.ones((self.number_of_particles,))
        
        def p_manufactured(t):
            return jnp.cos(t) * jnp.ones((self.number_of_particles,))
        
        def dt_p_manufactured(t):
            return -jnp.sin(t) * jnp.ones((self.number_of_particles,))
        
        def manufactured_solution(t):
            # t is scalar, stack to shape (2*number_of_particles,)
            return jnp.hstack((q_manufactured(t), p_manufactured(t)))
        
        def g_manufactured_solution(t):
            # t is scalar
            u = control_manufactured_solution(t)
            qp = jnp.hstack((q_manufactured(t), p_manufactured(t)))
            qp_dot = jnp.hstack((dt_q_manufactured(t), dt_p_manufactured(t)))
            e = self.nabla_2_ham(None, qp)
            rhs = (self.J_matrix - self.R_matrix) @ e + self.B(u)
            return qp_dot - rhs
            
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