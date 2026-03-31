#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements the proposed passive nonlinear controller
# and can be used to simulate the closed loop interaction of the controller
# and a given system.
#
#

# jax
import jax
import jax.numpy as jnp

# scipy for ARE solver and sparse matrices
from scipy.linalg import solve_continuous_are
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array

# helpers
from helpers.gauss import gauss_points_and_weights, gauss_quadrature_with_values
from helpers.legendre import scaled_legendre
from helpers.newton import newton_lineax
from main.time_discretization import implicit_midpoint, discrete_gradient
from helpers.nonlinear_system import NonlinearAffineSystem
from main.optimal_control import OptimalControlProblem
from helpers.visualization import plot_controller_trajectory, plot_coupled_trajectory
from helpers.other import vmap2d, style

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
import numpy as np

# defaults for optimization
OCP_NT = 1000
OCP_CUTOFF_TIME = 20.0
OCP_CUTOFF_PENALIZE = True
OCP_MAX_ITER = 40
OCP_DEBUG = True # TODO - set to False

# defaults for outer simulation
SIMULATION_NT = 500
SIMULATION_FINAL_TIME = 10.0
SIMULATION_DEBUG = True # TODO

# defaults for policy iteration
# space discretization
# POL_XLIMS = jnp.array([-0.5, 0.5])
POL_XLIMS = jnp.array([-2., 2.])
# POL_XLIMS = jnp.array([-4., 4.])
# POL_YLIMS = jnp.array([-0.5, 0.5])
POL_YLIMS = jnp.array([-2., 2.])
POL_DEGREE_X = 10
POL_DEGREE_Y = 10
# main iteration
POL_MAX_ITER = 50
POL_CHANGE_TOL = 1e-14
POL_RELATIVE_CHANGE_TOL = 1e-10
POL_USE_STOPPING_CRITERON = True
POL_DEBUG = True # TODO

# setup default options dictionary
SIMULATION_OPTIONS = {
    'ocp_nt': OCP_NT,
    'ocp_cutoff_time': OCP_CUTOFF_TIME,
    'ocp_cutoff_penalize': OCP_CUTOFF_PENALIZE,
    'ocp_max_iter': OCP_MAX_ITER,
    'ocp_debug': OCP_DEBUG,
    #
    'simulation_nt': SIMULATION_NT,
    'simulation_final_time': SIMULATION_FINAL_TIME,
    'controller_integrator': 'discrete gradient',
    'coupled_integrator': 'implicit midpoint',
    'controller_eta_inv_newton_steps': 10,
    'controller_eta_inv_newton_debug': True, # TODO
    'simulation_debug': SIMULATION_DEBUG,
    #
    'pol_degree_x': POL_DEGREE_X,
    'pol_degree_y': POL_DEGREE_Y,
    'pol_xlims': POL_XLIMS,
    'pol_ylims': POL_YLIMS,
    'pol_max_iter': POL_MAX_ITER,
    'pol_change_tol': POL_CHANGE_TOL,
    'pol_relative_change_tol': POL_RELATIVE_CHANGE_TOL,
    'pol_use_stopping_criterion': POL_USE_STOPPING_CRITERON,
    'pol_debug': POL_DEBUG,
    }


class Controller:

    def __init__(
            self,
            plant: NonlinearAffineSystem,
            controller_initial_state: jnp.ndarray = None,
            ocp_initial_control_routine: callable = None,
            use_policy_iteration: bool = True,
            value_function_savepath: str = None,
            options: dict = SIMULATION_OPTIONS,
            ):
        """
        sets up the passive nonlinear controller for the plant

        d/dt z = f(z) + B(z) u                         (1)
        y = h(z)

        given by

        d/dt zhat = f(zhat)
                    - B(zhat) B(zhat)^T etahat(zhat)
                    - B(zhat) h(zhat)
                    + B(zhat) v
        yhat = B(zhat)^T etahat(zhat)

        where etahat(zhat) is the gradient of the value function of

        min_{u} 1/2 * int_0^inf h(z)^T h(z) + u^T u dt    (2)

        subject to the plant (1) with initial condition z(0) = zhat.

        etahat(zhat) is given by the optimal adjoint evaluated at 0.

        in the implementation, a cutoff time T < inf is used to
        approximate the integral. depending on the value of
        penalize_cutoff, this cutoff is penalized by a quadratic
        approximation of the value function at the cutoff time
        or not.

        Args:
            plant:
            controller_initial_state: (optional) initial state of the controller
            ocp_initial_control_routine: (optional) callable that returns the initial control for the optimal control problem, given an initial state
            use_policy_iteration: (optional) whether to use policy iteration to obtain the value function and gradient of the value function
        """

        # plant properties
        self.plant = plant

        # other properties
        self.ocp_nt = options.get('ocp_nt', SIMULATION_OPTIONS['ocp_nt'])
        self.ocp_cutoff_time = options.get('ocp_cutoff_time', SIMULATION_OPTIONS['ocp_cutoff_time'])
        self.ocp_cutoff_penalize = options.get('ocp_cutoff_penalize', SIMULATION_OPTIONS['ocp_cutoff_penalize'])
        self.ocp_max_iter = options.get('ocp_max_iter', SIMULATION_OPTIONS['ocp_max_iter'])
        self.ocp_debug = options.get('ocp_debug', SIMULATION_OPTIONS['ocp_debug'])
        self.use_policy_iteration = use_policy_iteration

        # initialize optimal control guess
        self.ocp_initial_control_routine = ocp_initial_control_routine
        if self.ocp_initial_control_routine is None:
            self.ocp_initial_control_routine = lambda z0: jnp.ones((self.ocp_nt, self.plant.ncontrol))
        # self.last_optimal_control = self.ocp_initial_control
        # self.last_optimal_control jnp.ones((self.ocp.nt, self.plant.ncontrol))

        # linearize around zero for cutoff penalization and initial policy
        linearization_point = jnp.zeros((self.plant.nsys,))
        self.A = jax.jacobian(self.plant.f)(linearization_point)
        self.B = self.plant.g(linearization_point)
        self.C = jax.jacobian(self.plant.h)(linearization_point)
        # solve control ARE
        cARE_solution = solve_continuous_are(a=self.A, b=self.B, q=self.C.T@self.C, r=jnp.eye(self.plant.ncontrol)) # scipy method
        self.cARE_solution = jnp.array(cARE_solution)
        # linearized value function
        self.linearized_value_function = lambda z: 1/2 * z.T @ self.cARE_solution @ z
        self.linearized_value_function_vmap = vmap2d(self.linearized_value_function)
        # gradient of linearized value function
        self.linearized_gradient_value_function = lambda z: self.cARE_solution @ z
        self.linearized_gradient_value_function_vmap = vmap2d(self.linearized_gradient_value_function)
        # linearized control policy, optimal feedback for linearized plant is u = -B^T @ self.cARE_solution @ z
        self.linearized_policy = lambda z: - self.B.T @ self.cARE_solution @ z
        self.linearized_policy_vmap = vmap2d(self.linearized_policy)

        # compute cutoff penalization
        if self.ocp_cutoff_penalize:
            # value function at cutoff time is approximated by
            #     V(z(T)) \approx 1/2 z(T)^T self.cARE_solution z(T)
            self.ocp_terminal_cost = lambda z: 1/2 * z.T @ self.cARE_solution @ z

        else:
            self.ocp_terminal_cost = lambda z: 0.0

        # set up optimal control problem
        self.ocp = OptimalControlProblem(
            system = self.plant,
            running_cost = self.ocp_running_cost,
            final_time = self.ocp_cutoff_time,
            terminal_cost = self.ocp_terminal_cost,
            initial_control = self.ocp_initial_control_routine(self.plant.initial_state),
            options={
                'nt': self.ocp_nt,
                'debug': self.ocp_debug,
                'max_iter': self.ocp_max_iter,
                'use_stopping_criterion': False,
                },
            )

        # integrator for controller dynamics - default is discrete gradient method
        self.controller_integrator = options.get('controller_integrator', SIMULATION_OPTIONS['controller_integrator'])

        # set up newton method for controller dynamcis and discrete gradient method
        self.controller_eta_inv_newton_steps = options.get('controller_eta_inv_newton_steps', SIMULATION_OPTIONS['controller_eta_inv_newton_steps'])
        self.controller_eta_inv_newton_debug = options.get('controller_eta_inv_newton_debug', SIMULATION_OPTIONS['controller_eta_inv_newton_debug'])
        if self.use_policy_iteration:
            self.eta_inv_rootfunction = lambda zhat, etahat: self.gradient_value_function(zhat) - etahat
        else:
            self.eta_inv_rootfunction = lambda zhat, etahat: self.get_etahat(self.get_ocp_result(zhat)) - etahat
        self.eta_inv_newton_solver = newton_lineax(
            self.eta_inv_rootfunction, max_iter=self.controller_eta_inv_newton_steps,
            debug=self.controller_eta_inv_newton_debug
            )
        self.eta_inv_newton_initial_guess = jnp.zeros((self.plant.nsys,))

        # integrator for coupled dynamics - default is implicit midpoint method, since plant may not be pH
        self.coupled_integrator = options.get('coupled_integrator', SIMULATION_OPTIONS['coupled_integrator'])

        if controller_initial_state is None:
            self.controller_initial_state = jnp.zeros((self.plant.nsys,))
        else:
            self.controller_initial_state = controller_initial_state

        self.simulation_nt = options.get('simulation_nt', SIMULATION_OPTIONS['simulation_nt'])
        self.simulation_final_time = options.get('simulation_final_time', SIMULATION_OPTIONS['simulation_final_time'])
        self.simulation_tt = jnp.linspace(0, self.simulation_final_time, self.simulation_nt)
        self.simulation_debug = options.get('simulation_debug', SIMULATION_OPTIONS['simulation_debug'])

        ### policy iteration
        # ansatz functions
        self.degree_x = options.get('pol_degree_x', SIMULATION_OPTIONS['pol_degree_x'])
        self.degree_y = options.get('pol_degree_y', SIMULATION_OPTIONS['pol_degree_y'])
        assert self.degree_x == self.degree_y, 'pol_degree_x and pol_degree_y must be equal'
        self.legendre_degree = self.degree_x

        # space discretization
        self.xlims = options.get('pol_xlims', SIMULATION_OPTIONS['pol_xlims'])
        self.ylims = options.get('pol_ylims', SIMULATION_OPTIONS['pol_ylims'])
        # scaling matrix and midpoint shift from [-1,1] to [a,b]
        self.scale_from_unit_square = jnp.diag(jnp.array([(self.xlims[1] - self.xlims[0])/2, (self.ylims[1]-self.ylims[0])/2]))
        self.scale_to_unit_square = jnp.diag(jnp.array([2/(self.xlims[1] - self.xlims[0]), 2/(self.ylims[1]-self.ylims[0])]))
        self.midpoint_shift = jnp.array([(self.xlims[0] + self.xlims[1])/2, (self.ylims[0] + self.ylims[1])/2])

        # gauss quadrature
        self.num_gauss_points = self.legendre_degree + 1
        self.gauss_points, self.gauss_weights = gauss_points_and_weights(self.num_gauss_points)
        self.gauss_X, self.gauss_Y = jnp.meshgrid(self.gauss_points, self.gauss_points, indexing='ij')
        self.gauss_XY = jnp.stack((self.gauss_X, self.gauss_Y), axis=2) # shape (ngauss, ngauss, 2)
        self.gauss_XY_scaled = jnp.einsum('nm, xym -> xyn', self.scale_from_unit_square, self.gauss_XY) + self.midpoint_shift # scaled and shifted to Omega

        # fine grid for convergence test and plotting
        self.num_fine_points = 100
        self.fine_x = jnp.linspace(-1., 1., self.num_fine_points)
        self.fine_y = jnp.linspace(-1., 1., self.num_fine_points)
        self.fine_X, self.fine_Y = jnp.meshgrid(self.fine_x, self.fine_y, indexing='ij')
        self.fine_XY = jnp.stack((self.fine_X, self.fine_Y), axis=2) # shape (npoints, npoints, 2)
        self.fine_XY_scaled = jnp.einsum('nm, xym -> xyn', self.scale_from_unit_square, self.fine_XY) + self.midpoint_shift # scaled and shifted to Omega
        self.fine_x_scaled = self.scale_from_unit_square[0,0] * self.fine_x + self.midpoint_shift[0] # scaled and shifted to [xlims[0], xlims[1]]
        self.fine_y_scaled = self.scale_from_unit_square[1,1] * self.fine_y + self.midpoint_shift[1] # scaled and shifted to [ylims[0], ylims[1]]

        # legendre polynomials
        self.legendre_gauss, self.derivative_legendre_gauss = scaled_legendre(self.legendre_degree, self.gauss_points)
        self.legendre_zero, _ = scaled_legendre(self.legendre_degree, jnp.array([0.])) # for constant term

        # tensor basis values at gauss points and zero
        self.psi_gauss = jnp.einsum('ix, jy -> ijxy', self.legendre_gauss, self.legendre_gauss)
        self.psi_zero = jnp.einsum('ix, jy -> ij', self.legendre_zero, self.legendre_zero)
        self.nabla_psi_gauss = jnp.stack(
            (
                jnp.einsum('ix, jy -> ijxy', self.derivative_legendre_gauss, self.legendre_gauss),
                jnp.einsum('ix, jy -> ijxy', self.legendre_gauss, self.derivative_legendre_gauss),
            ),
            axis=-1,
            ) # shape (degree+1, degree+1, ngauss, 2)

        # function values at shifted and scaled gauss points. this introduces a chain rule factor in the derivative, see below
        self.f_gauss = vmap2d(self.plant.f)(self.gauss_XY_scaled)
        self.g_gauss = vmap2d(self.plant.g)(self.gauss_XY_scaled)
        self.h_gauss = vmap2d(self.plant.h)(self.gauss_XY_scaled)
        self.h_h_gauss = jnp.einsum('xyp, xyp -> xy', self.h_gauss, self.h_gauss)

        # policy iteration parameters
        self.pol_max_iter = options.get('pol_max_iter', SIMULATION_OPTIONS['pol_max_iter'])
        self.pol_change_tol = options.get('pol_change_tol', SIMULATION_OPTIONS['pol_change_tol'])
        self.pol_relative_change_tol = options.get('pol_relative_change_tol', SIMULATION_OPTIONS['pol_relative_change_tol'])
        self.pol_debug = options.get('pol_debug', SIMULATION_OPTIONS['pol_debug'])

        # perform policy iteration if wanted
        if self.use_policy_iteration:
            assert self.plant.nsys == 2, 'policy iteration is only implemented for 2D systems'
            if self.simulation_debug: print(f'{style.info}{style.bold}[controller] performing policy iteration{style.end}\n')
            self.policy_iteration() # makes self.value_function, self.gradient_value_function, self.optimal_policy available
            if self.pol_debug or (value_function_savepath is not None): # TODO: remove?
                self.plot_policy_results(
                    show_value_function = True,
                    show_gradient = False,
                    show_linearized_value_function = False,
                    show_difference = False,
                    interpolation='lanczos',
                    savepath = value_function_savepath,
                    )
                # self.compare_policy_and_ocp()


        # extended kalman filter settings
        # weighting matrices
        self.R_ekf = jnp.eye(self.plant.ncontrol)
        self.R_ekf_inv = jnp.linalg.inv(self.R_ekf)
        self.Q_ekf = jnp.eye(self.plant.nsys)

        # initial condition for covariance
        self.Pi_ekf_0 = jnp.eye(self.plant.nsys)

        # derivatives
        self.f_ekf = lambda z_ekf: self.plant.f(z_ekf) - self.plant.g(z_ekf) @ self.get_yhat(z_ekf)
        self.F_ekf = jax.jacobian(self.f_ekf)
        self.H_ekf = jax.jacobian(self.plant.h)

        return

    def ocp_running_cost(
            self,
            z: jnp.ndarray,
            u: jnp.ndarray,
            ):

        hh = self.plant.h(z)

        return 1/2 * (hh.T @ hh + u.T @ u)

    ### policy iteration
    def pol_hjb_solver(
            self,
            ):

        # evaluate current policy at gauss points
        self.u_gauss = self.current_policy_vmap(self.gauss_XY_scaled) # shape (ngauss, ngauss, ncontrol)

        g_u_gauss = jnp.einsum('xynm, xym -> xyn', self.g_gauss, self.u_gauss)

        # after the scalings, the functions f, g, h map from [-1,1] to R^2
        # this introduces as chain rule factor in the derivative given by
        #
        #   scale_from_unit_square @ dt z_scaled = f_scaled(z_scaled) + g_scaled(z_scaled) @ u_scaled
        #
        # hence, f_scaled + g_scaled @ u_scaled = scale_to_unit_square @ (dt z_scaled).
        # this scaling matrix needs to be accounted for in the HJB equation!

        scaled_f_g_u_gauss = jnp.einsum('nm, xym -> xyn', self.scale_to_unit_square, self.f_gauss + g_u_gauss)

        lhs_gauss = jnp.einsum('ijxyn, xyn, kmxy -> kmijxy', self.nabla_psi_gauss, scaled_f_g_u_gauss, self.psi_gauss)

        # integrate in both directions to get M
        M = gauss_quadrature_with_values(
            self.gauss_weights,
            gauss_quadrature_with_values(
                self.gauss_weights,
                lhs_gauss,
                axis=-1,
                interval = (-1., 1.)
                ),
            axis=-1,
            interval = (-1., 1.)
            )

        # make M invertible
        M = M.at[0,0,0,0].set(1.)

        # compute coefficient vector
        # build coefficient vector
        self.u_u_gauss = jnp.einsum('xym, xym -> xy', self.u_gauss, self.u_gauss)

        # right hand side vector - integrate in both directions
        c = gauss_quadrature_with_values(
            self.gauss_weights,
            gauss_quadrature_with_values(
                self.gauss_weights,
                jnp.einsum('xy, kmxy -> kmxy', 1/2 * (self.h_h_gauss + self.u_u_gauss), self.psi_gauss),
                axis = -1,
                interval = (-1., 1.)
                ),
            axis = -1,
            interval = (-1., 1.)
            )

        # reshape to make it a matrix equation
        dp1 = self.legendre_degree + 1

        M_reshaped = M.reshape((dp1**2, dp1**2))
        c_reshaped = c.reshape((dp1**2,))

        # solve M alpha = - c
        coefficients = jnp.linalg.solve(M_reshaped, - c_reshaped).reshape((dp1, dp1))

        # fix constant term
        computed_value_at_zero = jnp.einsum('ij, ij -> ', coefficients, self.psi_zero)
        constant_polynomial_value = self.psi_zero[0,0]
        correction = - 1/constant_polynomial_value * computed_value_at_zero
        coefficients = coefficients.at[0,0].set(correction)

        # these coefficients can be used to evaluate the value function at any point
        # by evaluating the polynomial at the scaled and shifted point. example:
        # value_function_gauss = jnp.einsum('ij, ijxy -> xy', coefficients, self.psi_gauss)

        return coefficients

    def get_value_function_and_policy(
            self,
            ):
        """
        TODO: documentation

        Returns:

        """

        def value_function_and_derivative(
                z: jnp.ndarray,
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
            """
            evaluates the value function parameterized by coeffs at the point z

            Args:
                z: point to evaluate at
                coeffs: coefficients of the value function in the basis psi_ij (shape (degree+1, degree+1))

            Returns:
                value function and its derivative at z
            """

            # z needs to be a vector of shape (nsys,)

            # scale z to unit square
            z = self.scale_to_unit_square @ (z - self.midpoint_shift)

            # evaluate psi_i at scaled z
            z_x, z_y = z[0], z[1]
            values_z_x, derivative_values_z_x = scaled_legendre(self.legendre_degree, z_x)
            values_z_y, derivative_values_z_y = scaled_legendre(self.legendre_degree, z_y)

            # get psi_i values at z_scaled
            psi_values_z = jnp.einsum('i, j -> ij', values_z_x, values_z_y) # shape (degree+1, degree+1)
            nabla_psi_values_z = jnp.stack(
                (
                    jnp.einsum('i, j -> ij', derivative_values_z_x, values_z_y),
                    jnp.einsum('i, j -> ij', values_z_x, derivative_values_z_y),
                ),
                axis=-1,
                ) # shape (degree+1, degree+1)

            value_function_z = jnp.einsum('ij, ij -> ', self.value_function_coefficients, psi_values_z)
            derivative_value_function_z = jnp.einsum('ij, nm, ijm -> n', self.value_function_coefficients, self.scale_to_unit_square, nabla_psi_values_z) # chain rule factor

            return value_function_z, derivative_value_function_z

        return (
            lambda z: value_function_and_derivative(z)[0], # value function
            lambda z: value_function_and_derivative(z)[1], # gradient of value function
            lambda z: - self.plant.g(z).T @ value_function_and_derivative(z)[1] # optimal policy
            )

    def policy_iteration(
            self,
            ):

        self.current_policy = self.linearized_policy # not really needed ... TODO: remove?
        self.current_policy_vmap = self.linearized_policy_vmap

        # evaluate initial policy on fine grid
        old_u_fine = self.current_policy_vmap(self.fine_XY_scaled)

        policy_iteration_converged = False

        for i in range(self.pol_max_iter):

            # solve hjb equation
            self.value_function_coefficients = self.pol_hjb_solver()

            # compute new policy
            self.value_function, self.gradient_value_function, self.current_policy = self.get_value_function_and_policy()

            # vmap new policy
            self.current_policy_vmap = vmap2d(self.current_policy)

            # check for convergence
            u_fine = self.current_policy_vmap(self.fine_XY_scaled) # evaluate new policy on fine grid
            max_diff_u_fine = jnp.max(jnp.abs(old_u_fine - u_fine))
            max_relative_diff_u_fine = max_diff_u_fine / jnp.max(jnp.abs(old_u_fine))
            if self.pol_debug:
                print(f'{style.info}[policy iteration] max absolute difference in policy = {max_diff_u_fine:.2e}{style.end}')
                print(f'{style.info}[policy iteration] max relative difference in policy = {max_relative_diff_u_fine:.2e}{style.end}')

            if max_diff_u_fine < self.pol_change_tol:
                if self.pol_debug: print(f'{style.success}{style.bold}[policy iteration] converged (absolute change tolerance met){style.end}')
                policy_iteration_converged = True
                break

            if max_relative_diff_u_fine < self.pol_relative_change_tol:
                if self.pol_debug: print(f'{style.success}{style.bold}[policy iteration] converged (relative change tolerance met){style.end}\n')
                policy_iteration_converged = True
                break

            # store values of policy on fine grid for convergence check
            old_u_fine = u_fine

            # print info
            if self.pol_debug: print(f'{style.info}{style.bold}[policy iteration] iteration {i+1} complete{style.end}')

        assert policy_iteration_converged, 'policy iteration did not converge'

        # # one last solve of HJB equation
        # # self.value_function_inner = self.hjb_solver(self.initial_guess_value_function_inner.reshape((-1,))) # version with helpers.newton solver
        # self.value_function_inner = self.hjb_solver() # version with spsolve

        self.optimal_policy = self.current_policy

        # make vmapped value function and gradient value function available
        self.value_function_vmap = vmap2d(self.value_function)
        self.gradient_value_function_vmap = vmap2d(self.gradient_value_function)

        # return callables
        return self.value_function, self.gradient_value_function, self.optimal_policy

    ### controller simulation
    def get_ocp_result(
            self,
            zhat: jnp.ndarray,
            ) -> dict:

        self.ocp.initial_state = zhat
        self.ocp.initial_control = self.ocp_initial_control_routine(zhat)
        # jax.debug.print(f'{style.warning}norm of initial state for gradient descent = {{x}}{style.end}', x=jnp.linalg.norm(zhat))
        # jax.debug.print(f'{style.warning}initial control for gradient descent = {{x}}{style.end}', x=self.ocp.initial_control)

        # solve optimal control problem
        ocp_result = self.ocp.gradient_descent()

        return ocp_result

    def get_hamiltonianhat(
            self,
            ocp_result: dict,
            ) -> jnp.ndarray:

        hamiltonianhat = ocp_result['optimal_cost']

        return hamiltonianhat

    def get_etahat(
            self,
            ocp_result: dict,
            ) -> jnp.ndarray:

        etahat = ocp_result['optimal_adjoint'][0,:] # at time 0

        return etahat

    def get_yhat(
            self,
            zhat: jnp.ndarray,
            ):

        if self.use_policy_iteration:
            # extract etahat from policy iteration result
            etahat = self.gradient_value_function(zhat)

        else:
            # solve optimal control problem and extract etahat
            ocp_result = self.get_ocp_result(zhat)
            etahat = self.get_etahat(ocp_result)

        yhat = self.plant.g(zhat).T @ etahat

        return yhat

    # version for implicit midpoint method
    def controller_f(
            self,
            zhat: jnp.ndarray,
            ):

        if self.use_policy_iteration:
            # extract etahat from policy iteration result
            etahat = self.gradient_value_function(zhat)

        else:
            # solve optimal control problem and extract etahat
            ocp_result = self.get_ocp_result(zhat)
            etahat = self.get_etahat(ocp_result)

        # compute fhat
        ghat = self.plant.g(zhat)
        fhat = self.plant.f(zhat) - ghat @ ghat.T @ etahat - ghat @ self.plant.h(zhat)

        return fhat

    # version for discrete gradient method
    def controller_r(
            self,
            etahat: jnp.ndarray, # gradient of the hamiltonian
            ):

        # compute zhat with newton solver
        zhat = self.eta_inv_newton_solver(self.eta_inv_newton_initial_guess, etahat)

        # compute fhat
        ghat = self.plant.g(zhat)
        fhat = self.plant.f(zhat) - ghat @ ghat.T @ etahat - ghat @ self.plant.h(zhat)

        return - fhat

    def controller_ham_eta(
            self,
            zhat: jnp.ndarray, # state of the controller
            ) -> (jnp.ndarray, jnp.ndarray):
        # returns (hamiltonianhat, etahat) as needed for discrete gradient method

        if self.use_policy_iteration:
            # extract hamiltonianhat and etahat from policy iteration result
            hamiltonianhat = self.value_function(zhat)
            etahat = self.gradient_value_function(zhat)

        else:
            # solve optimal control problem and extract hamiltonianhat and etahat
            ocp_result = self.get_ocp_result(zhat)
            hamiltonianhat = self.get_hamiltonianhat(ocp_result)
            etahat = self.get_etahat(ocp_result)

        # compute fhat
        # ghat = self.plant.g(zhat)
        # fhat = self.plant.f(zhat) - ghat @ ghat.T @ etahat - ghat @ self.plant.h(zhat)

        return hamiltonianhat, etahat

    def simulate_controller_trajectory(
            self,
            control: callable = None,
            title: str = None,
            integration_method: str = None,
            ):

        if control is None:
            control = lambda u: jnp.zeros((self.plant.ncontrol,))

        if integration_method is None:
            integration_method = self.controller_integrator

        # initial state
        initial_zhat = self.controller_initial_state

        # integrate
        if integration_method == 'discrete gradient':
            controller_trajectory_and_hamiltonian = discrete_gradient(
                self.controller_r,
                self.controller_ham_eta,
                self.plant.g,
                self.simulation_tt,
                initial_zhat,
                control(self.simulation_tt),
                debug = self.simulation_debug,
                return_hamiltonian = True,
                )

            controller_trajectory, controller_hamiltonian = controller_trajectory_and_hamiltonian

        else: # fall back to implicit midpoint method
            controller_trajectory = implicit_midpoint(
                lambda zhat, uhat: self.controller_f(zhat) + self.plant.g(zhat) @ uhat,
                self.simulation_tt,
                self.controller_initial_state,
                control(self.simulation_tt),
                debug = self.simulation_debug,
                )

            controller_hamiltonian = jnp.zeros((self.simulation_nt,))

            # calculate hamiltonian values
            if self.use_policy_iteration:
                controller_hamiltonian = jax.vmap(self.value_function)(controller_trajectory)
                # controller_hamiltonian = self.value_function_vmap(controller_trajectory)
            else:
                # very inefficient, but easy to implement
                for i in range(self.simulation_nt):
                    ham_zi = self.get_hamiltonianhat(self.get_ocp_result(controller_trajectory[i,:]))
                    controller_hamiltonian = controller_hamiltonian.at[i].set(ham_zi)

            controller_trajectory_and_hamiltonian = (controller_trajectory, controller_hamiltonian)

        if title is not None:
            plot_controller_trajectory(
                controller_trajectory = controller_trajectory,
                tt = self.simulation_tt,
                title=title,
                )

        return controller_trajectory_and_hamiltonian

    def ekf_dynamics(self, ekf_state, u_ekf):

        # extract state and covariance
        z_ekf = ekf_state[:self.plant.nsys]
        Pi_ekf = ekf_state[self.plant.nsys:].reshape((self.plant.nsys, self.plant.nsys))

        # observer gain
        K_ekf = Pi_ekf @ self.H_ekf(z_ekf).T @ self.R_ekf_inv

        # state update
        d_z_ekf = self.f_ekf(z_ekf) + K_ekf @ (u_ekf - self.plant.h(z_ekf))

        # covariance update
        d_Pi_ekf = self.F_ekf(z_ekf) @ Pi_ekf + Pi_ekf @ self.F_ekf(z_ekf).T - K_ekf @ self.R_ekf @ K_ekf.T + self.Q_ekf

        return jnp.hstack((d_z_ekf, d_Pi_ekf.reshape((self.plant.nsys**2,))))

    def simulate_ekf_trajectory(
            self,
            control: callable = None,
            title: str = None,
            ):
        """
        simulates the extended kalman filter

        """

        if control is None:
            control = lambda u: jnp.zeros((self.plant.ncontrol,))

        ekf_initial_condition = jnp.hstack((self.controller_initial_state, self.Pi_ekf_0.reshape((self.plant.nsys**2,))))

        # integrate with implicit midpoint method
        ekf_controller_trajectory = implicit_midpoint(
            self.ekf_dynamics,
            self.simulation_tt,
            ekf_initial_condition,
            control(self.simulation_tt),
            debug = self.simulation_debug,
            )

        # remove covariance from trajectory
        ekf_controller_trajectory = ekf_controller_trajectory[:,:self.plant.nsys]

        ekf_controller_hamiltonian = jnp.zeros((self.simulation_nt,))

        # calculate hamiltonian values
        if self.use_policy_iteration:
            ekf_controller_hamiltonian = jax.vmap(self.value_function)(ekf_controller_trajectory)
            # controller_hamiltonian = self.value_function_vmap(controller_trajectory)
        else:
            # very inefficient, but easy to implement
            for i in range(self.simulation_nt):
                ham_zi = self.get_hamiltonianhat(self.get_ocp_result(ekf_controller_trajectory[i,:]))
                ekf_controller_hamiltonian = ekf_controller_hamiltonian.at[i].set(ham_zi)

        ekf_controller_trajectory_and_hamiltonian = (ekf_controller_trajectory, ekf_controller_hamiltonian)

        if title is not None:
            plot_controller_trajectory(
                controller_trajectory = ekf_controller_trajectory,
                tt = self.simulation_tt,
                title=title,
                )

        return ekf_controller_trajectory_and_hamiltonian

    def coupled_dynamics(
            self,
            z_zhat: jnp.ndarray, # merged z and zhat
            u_uhat: jnp.ndarray, # merged controls, not used
            ) -> jnp.ndarray:
        """
        power conserving interconnection of the plant and the controller
        """

        z, zhat = z_zhat[:self.plant.nsys], z_zhat[self.plant.nsys:]

        if self.use_policy_iteration:
            # extract etahat from policy iteration result
            etahat = self.gradient_value_function(zhat)

        else:
            # solve optimal control problem and extract etahat
            ocp_result = self.get_ocp_result(zhat)
            etahat = self.get_etahat(ocp_result)

        dt_z = self.plant.f(z) - self.plant.g(z) @ self.get_yhat(zhat)

        Bzhat = self.plant.g(zhat)
        dt_zhat = (
                self.plant.f(zhat)
                - Bzhat @ Bzhat.T @ etahat
                - Bzhat @ self.plant.h(zhat) # up until here is fhat(zhat)
                + Bzhat @ self.plant.h(z)
                )

        return jnp.hstack((dt_z, dt_zhat))

    def simulate_coupled_trajectory(
            self,
            use_ekf: bool = False,
            title: str = None,
            ):

        # initial state and dynamics without extended kalman filter
        coupled_initial_state = jnp.hstack((self.plant.initial_state, self.controller_initial_state))
        coupled_dynamics = self.coupled_dynamics

        # overwrite for extended kalman filter
        if use_ekf:

            ekf_initial_condition = jnp.hstack((self.controller_initial_state, self.Pi_ekf_0.reshape((self.plant.nsys**2,))))

            coupled_initial_state = jnp.hstack((self.plant.initial_state, ekf_initial_condition))

            def coupled_dynamics(coupled_state, coupled_control):

                # extract state of plant and ekf
                z = coupled_state[:self.plant.nsys]
                z_ekf = coupled_state[self.plant.nsys:self.plant.nsys*2]
                ekf_state = coupled_state[self.plant.nsys:]

                # plant state update
                d_z = self.plant.f(z) - self.plant.g(z) @ self.get_yhat(z_ekf)

                # ekf state update
                d_ekf_state = self.ekf_dynamics(ekf_state, self.plant.h(z))

                return jnp.hstack((d_z, d_ekf_state))

        # integrate
        if self.coupled_integrator == 'implicit midpoint':
            coupled_trajectory = implicit_midpoint(
                coupled_dynamics,
                self.simulation_tt,
                coupled_initial_state,
                jnp.zeros((self.simulation_nt, self.plant.ncontrol)),
                debug = self.simulation_debug,
                )
        else:
            raise NotImplementedError('only implicit midpoint method is implemented for coupled dynamics')

        # if ekf is used, get rid of covariance matrix components in the state
        if use_ekf:
            coupled_trajectory = coupled_trajectory[:, :self.plant.nsys*2]

        if title is not None:
            plot_coupled_trajectory(
                coupled_trajectory = coupled_trajectory,
                tt = self.simulation_tt,
                title = title,
            )

        return coupled_trajectory

    def plot_policy_results(
            self,
            show_value_function: bool = True,
            show_gradient: bool = True,
            show_linearized_value_function: bool = False,
            show_difference: bool = False,
            interpolation: str = 'nearest',
            savepath: str = None,
            ):

        value_function_fine = self.value_function_vmap(self.fine_XY_scaled)
        linearized_value_function_fine = self.linearized_value_function_vmap(self.fine_XY_scaled)
        diff_fine = jnp.abs(value_function_fine - linearized_value_function_fine)
        relative_diff_fine = diff_fine / jnp.max(jnp.abs(linearized_value_function_fine))

        # compute gradient data
        gradient_data = self.gradient_value_function_vmap(self.fine_XY_scaled)
        gradient_data_x = np.array(gradient_data[:,:,0])
        gradient_data_y = np.array(gradient_data[:,:,1])
        gradient_norm = jnp.linalg.norm(gradient_data, axis=-1)
        gradient_magnitude = 2*np.array(gradient_norm / jnp.max(gradient_norm)) # for visualization

        if show_value_function:
            # transpose is needed because of weird imshow convention
            plt.imshow(value_function_fine.T, origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)),)
            cbar = plt.colorbar()
            # cbar.ax.get_yaxis().labelpad = 20
            cbar.set_label(r'$\hat{\mathcal{H}}(z)$')
            if show_gradient:
                # transposing is needed to fix the effect of indexing='ij' in meshgrid
                plt.streamplot(np.array(self.fine_x_scaled), np.array(self.fine_y_scaled), gradient_data_x.T, gradient_data_y.T, color='white', linewidth=gradient_magnitude, density=0.4, arrowstyle='wedge')
            plt.contour(self.fine_x_scaled, self.fine_y_scaled, value_function_fine.T, levels=5, colors='white', alpha=0.2)
            plt.axis('equal')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath + '_value_function.png')
                plt.savefig(savepath + '_value_function.pgf')
                print(f'{style.success}saved value function plot to {savepath} (as .pgf and .png){style.end}')
            plt.title('value function')
            plt.tight_layout()
            plt.show()

        if show_gradient and not show_value_function:
            # transposing is needed to fix the effect of indexing='ij' in meshgrid
            streamplot = plt.streamplot(np.array(self.fine_x_scaled), np.array(self.fine_y_scaled), gradient_data_x.T, gradient_data_y.T, color=gradient_magnitude, cmap='viridis', linewidth=gradient_magnitude)
            plt.colorbar(streamplot.lines)
            plt.xlim(self.xlims[0], self.xlims[1])
            plt.ylim(self.ylims[0], self.ylims[1])
            plt.title('gradient of value function')
            plt.axis('equal')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.tight_layout()
            plt.show()

        if show_linearized_value_function:
            # transpose is needed because of weird imshow convention
            plt.imshow(linearized_value_function_fine.T, origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)))
            plt.colorbar()
            plt.contour(self.fine_x_scaled, self.fine_y_scaled, linearized_value_function_fine.T, levels=5, colors='white', alpha=0.2)
            plt.axis('equal')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath + '_linearized_value_function.png')
                plt.savefig(savepath + '_linearized_value_function.pgf')
                print(f'{style.success}saved linearized value function plot to {savepath} (as .pgf and .png){style.end}')
            plt.title('value function of linearized plant')
            plt.tight_layout()
            plt.show()

        if show_difference:
            # transpose is needed because of weird imshow convention
            plt.imshow(relative_diff_fine.T, origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)))
            plt.colorbar()
            plt.axis('equal')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath + '_difference_value_function.png')
                plt.savefig(savepath + '_difference_value_function.pgf')
                print(f'{style.success}saved difference value function plot to {savepath} (as .pgf and .png){style.end}')
            plt.title('relative difference of value function and linearized value function')
            plt.tight_layout()
            plt.show()

        return

    def compare_policy_and_ocp(
            self,
            xsamples: int = 20,
            ysamples: int = 20,
            interpolation: str = 'nearest',
            ):

        # sample points
        x = jnp.linspace(self.xlims[0], self.xlims[1], xsamples)
        y = jnp.linspace(self.ylims[0], self.ylims[1], ysamples)
        X, Y = jnp.meshgrid(x, y)
        XY = jnp.stack((X, Y), axis=2) # shape (xsamples, ysamples, 2)

        # vectorize ocp solver
        get_ocp_result_vmap = vmap2d(self.get_ocp_result)

        # ocp at sample points
        ocp_results = get_ocp_result_vmap(XY)
        ocp_vfun = ocp_results['optimal_cost']

        # policy at sample points
        policy_vfun = self.value_function_vmap(XY)

        # plot
        plt.imshow(ocp_vfun, origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)))
        plt.colorbar()
        plt.title('value function from ocp solver')
        plt.axis('equal')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.tight_layout()
        plt.show()

        plt.imshow(policy_vfun, origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)))
        plt.colorbar()
        plt.title('value function from policy iteration')
        plt.axis('equal')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.tight_layout()
        plt.show()

        plt.imshow((ocp_vfun - policy_vfun)/jnp.max(ocp_vfun), origin='lower', interpolation=interpolation, extent=jnp.hstack((self.xlims, self.ylims)))
        plt.colorbar()
        plt.title('relative difference')
        plt.axis('equal')
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.tight_layout()
        plt.show()

        return








if __name__ == "__main__":

    # tests are moved to tests/

    # jax config
    jax.config.update('jax_enable_x64', True)

    # plotting
    from helpers.other import mpl_settings
    mpl_settings()

    # examples
    from examples.lti import LTI
    from examples.pendulum import Pendulum
    from examples.van_der_pol import VanDerPol

    controller = Controller(
        # plant = LTI(),
        # plant = Pendulum(),
        plant = VanDerPol(),
        options = {
            'pol_degree_x': 15,
            'pol_degree_y': 15,
            'pol_xlims': jnp.array([-2.5, 2.5]),
            'pol_ylims': jnp.array([-2.5, 2.5]),
            }
        # use_policy_iteration = True, # True is the default
        )

    pass