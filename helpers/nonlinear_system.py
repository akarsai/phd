#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements an abstract class for nonlinear systems
#
#

# jax
import jax.numpy as jnp

class NonlinearSystem:

    def __init__(
            self,
            dynamics: callable,
            output: callable,
            initial_state: jnp.ndarray,
            ncontrol: int,
            nsys: int = None,
            ):
        """
        implementation of the dynamical system

        d/dt z = dynamics(z, u), z(0) = initial_state
        y = output(z, u)

        Args:
            dynamics: callable dynamics function
            dynamics: callable output function
            initial_state: initial state of the optimization problem
            ncontrol: number of control variables
            nsys: (optional) number of system variables
        """

        self.dynamics = dynamics
        self.output = output
        self.initial_state = initial_state
        self.ncontrol = ncontrol
        if nsys is None:
            self.nsys = initial_state.shape[0]
        else:
            self.nsys = nsys

        return

class NonlinearAffineSystem(NonlinearSystem):

    def __init__(
            self,
            f: callable,
            g: callable,
            h: callable,
            k: callable,
            initial_state: jnp.ndarray,
            ncontrol: int,
            nsys: int = None
            ):
        """
        implementation of the affine dynamical system

        d/dt z = f(z) + g(z) u, z(0) = initial_state
        y = h(z) + D(z) u

        Args:
            f: callable function describing the dynamics
            g: callable function describing control dependence
            h: callable function describing uncontrolled output
            k: callable function describing input feedthrough in output
            initial_state: initial state of the optimization problem
            ncontrol: number of control variables
            nsys: (optional) number of system variables
        """

        self.f = f
        self.g = g
        self.h = h
        self.k = k

        super().__init__(
            dynamics = lambda z, u: f(z) + g(z) @ u,
            output = lambda z, u: h(z) + k(z) @ u,
            initial_state = initial_state,
            ncontrol = ncontrol,
            nsys = nsys,
            )

        return

class NonlinearAffineSystem_NoFeedthrough(NonlinearAffineSystem):

    def __init__( self, *args, **kwargs ):
        """
        implementation of the affine dynamical system

        d/dt z = f(z) + g(z) u, z(0) = initial_state
        y = h(z)

        Args:
            f: callable function describing the dynamics
            B: callable function describing control dependence
            h: callable function describing uncontrolled output
            initial_state: initial state of the optimization problem
            ncontrol: number of control variables
            nsys: (optional) number of system variables
        """

        ncontrol = kwargs['ncontrol']

        super().__init__(*args, **kwargs, k = lambda z: jnp.zeros((ncontrol, ncontrol)))

        return