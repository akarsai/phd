#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the van der pol oscillator
#

# jax
import jax
import jax.numpy as jnp

# scipy ARE solve
from scipy.linalg import solve_continuous_are

# nonlinear system class
from helpers.nonlinear_system import NonlinearAffineSystem_NoFeedthrough

# time discretization
from main.time_discretization import implicit_midpoint

# printing
from helpers.other import style

class VanDerPol(NonlinearAffineSystem_NoFeedthrough):

    def __init__(
            self,
            initial_state: jnp.ndarray = None,
            mu: float = 2.0,
            damping: float = 2.2,
            observation: str = 'position',
            plot_limit_cycle: bool = False,
            ):
        """
        models the van der pol oscillator with friction and velocity control

        the dynamics read as

        ddot{x} = mu * (1 - x^2) * dot{x} - damping * dot{x} - x + u

        after order reduction, we arrive at

        d/dt z1 = z2
        d/dt z2 = mu * (1 - z1^2) * z2 - damping * z2 - z1 + u

        Args:
            initial_state: (optional) initial state of the oscillator
            mu: (optional) parameter mu
        """

        self.damping = damping

        def f(z):
            return jnp.array([
                z[1],
                mu * (1 - z[0]**2) * z[1] - self.damping * z[1] - z[0]
                ])

        # self.B = jnp.array([[0],[0.01]])
        self.B = jnp.array([[0],[1]])
        self.C = jnp.array([[1, 1]]) # only used for position observation

        if initial_state is None:
            initial_state = jnp.array([1.0, -0.5])

        # as the output operator, we take
        #   h(z) = dist(z, M),
        # where M is the limit cycle of the van der pol oscillator

        # sample limit cycle
        limit_cycle_final_time = 100.0
        limit_cycle_nt = 10000
        limit_cycle_tt = jnp.linspace(0.0, limit_cycle_final_time, limit_cycle_nt)
        limit_cycle_zz = implicit_midpoint(lambda z, u: f(z) + self.B @ u, limit_cycle_tt, initial_state, jnp.zeros((limit_cycle_nt, 1)))
        M = limit_cycle_zz[9000:,:] # extract points corresponding to time horizon [90, 100]
        M = M[::5,:] # sample down to 200 points
        # M = M[::10,:] # sample down to 100 points

        if plot_limit_cycle:
            import matplotlib.pyplot as plt
            from helpers.other import mpl_settings
            mpl_settings()
            plt.plot(M[:,0], M[:,1])
            plt.title(f'limit cycle of van der pol oscillator, damping = {self.damping}')
            plt.show()

        # distance function
        distance_to_limit_cycle = lambda z: jnp.min(jax.vmap(lambda p: jnp.linalg.norm(p - z))(M)).reshape((-1,))
        distance_to_limit_cycle = jax.jit(distance_to_limit_cycle)

        if observation == 'distance to limit cycle':
            print(f'\n{style.info}distance to limit cycle for van der pol oscillator{style.end}')
            h = distance_to_limit_cycle
        else:
            print(f'\n{style.info}position observation for van der pol oscillator{style.end}')
            h = lambda z: self.C @ z # position observation

        super().__init__(
            f = f,
            g = lambda z: self.B,
            h = h,
            initial_state = initial_state,
            ncontrol = 1
            )

        return

    def compute_ARE_solution(
            self,
            linearize_around: str = 'zero',
            ):

        # linearize around zero
        if linearize_around == 'zero':
            linearization_point = jnp.zeros((self.nsys,))
        else:
            linearization_point = self.initial_state
        A = jax.jacobian(self.f)(linearization_point)
        B = self.g(linearization_point)
        C = jax.jacobian(self.h)(linearization_point)

        # solve ARE
        X = solve_continuous_are(a=A, b=B, q=C.T@C, r=jnp.eye(self.ncontrol)) # scipy method
        X = jnp.array(X)

        return A, B, X


    def compute_stabilizing_control(
            self,
            A_B_X: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            z0: jnp.ndarray,
            final_time: float = 20.0,
            nt: int = 100,
            ):

        # A_B_X should be obtained from compute_ARE_solution
        A, B, X = A_B_X

        # optimal feedback law is u = - B^T X z
        # obtain optimal state / adjoint / control for time horizon [0, final_time]
        tt = jnp.linspace(0, final_time, nt)

        # nonlinear control
        zz = implicit_midpoint(lambda z, u: self.dynamics(z,u) - B @ B.T @ X @ z , tt, z0, jnp.zeros((nt, self.ncontrol)))
        uu = jnp.einsum('ab,tb->ta', - B.T @ X, zz)


        # linear control for comparison
        zz_lin = implicit_midpoint(lambda z, u: (A - B @ B.T @ X) @ z + B @ u , tt, z0, jnp.zeros((nt, self.ncontrol)))

        plt.plot(tt, zz, label=r'$u = -B^T X z_{\text{nonlin}}$')
        plt.legend()
        plt.title(f'van der pol oscillator reacting to control, damping = {vanderpol.damping}')
        plt.show()

        return uu

    
if __name__ == "__main__":
    
    pass