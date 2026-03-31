#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the hill-moylan system found in the
# reference
#
# Hill, D. and Moylan, P.
# Connections between finite-gain and asymptotic stability
# doi: 10.1109/TAC.1980.1102463
#

# jax
import jax.numpy as jnp

# nonlinear system class
from helpers.nonlinear_system import NonlinearAffineSystem

class HillMoylan(NonlinearAffineSystem):

    def __init__(
            self,
            initial_state = None,
            lam = 1.0,
            alpha = 2.0,
            ):
        """
        models the nonlinear  system

        d/dt z = - z - (alpha z)/(1+z^4) + 2 lam u
        y = (alpha z)/(1+z^4) + lam u

        found in the reference 10.1109/TAC.1980.1102463 .
        the system is dissipative with the quadratic supply

        s(u,y) = lam u^2 - y^2

        and a storage function is H(z) = alpha/2 arctan(z^2)

        Args:
            initial_state: (optional) initial state of the system
            lam: (optional) parameter lam in system dynamics
            alpha: (optional) parameter alpha in system dynamics
        """

        self.lam = lam
        self.alpha = alpha

        self.hamiltonian = lambda z: self.alpha/2 * jnp.arctan(z[0]**2)
        self.eta = lambda z: jnp.array([(self.alpha * z[0])/(1 + z[0]**4)])

        if initial_state is None:
            initial_state = jnp.array([1.0])

        super().__init__(
            f = lambda z: (- z[0] - (self.alpha * z[0])/(1 + z[0]**4)).reshape((1,)),
            g = lambda z: jnp.array([2 * self.lam]).reshape((1,1)),
            h = lambda z: (self.alpha * z[0])/(1 + z[0]**4),
            k = lambda z: jnp.array([self.lam]).reshape((1,1)),
            initial_state = initial_state,
            ncontrol = 1,
            )

        return


    
if __name__ == "__main__":
    
    pass