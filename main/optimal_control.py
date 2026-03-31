#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a class for optimal control problems.
# to solve the optimal control problems, either a primal-dual
# or a jax-autodiff based gradient calculation can be used.
#

# matplotlib
import matplotlib.pyplot as plt

# jax
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jit, vmap, random

# helpers
from main.time_discretization import implicit_midpoint, discrete_gradient
from helpers.other import style
from helpers.other import mpl_settings
from helpers.nonlinear_system import NonlinearSystem

# timing
from timeit import default_timer as timer

# defaults for gradient calculation
OCP_NT = 100
OCP_STATE_TIME_INTEGRATION = 'implicit midpoint' # 'implicit euler', 'bdf4'
OCP_ADJOINT_TIME_INTEGRATION = 'implicit midpoint' # 'implicit euler', 'bdf4'
OCP_GRADIENT_CALCULATION = 'primal dual' # if not 'jax', then primal dual is used

# defaults for gradient descent
OCP_MAX_ITER = 100
OCP_NORM_GRADIENT_TOL = 1e-14
OCP_RELATIVE_NORM_GRADIENT_TOL = 1e-14
OCP_USE_STOPPING_CRITERON = True

# default debug flag
OCP_DEBUG = True

# setup default options dictionary
OCP_OPTIONS = {
    'nt': OCP_NT,
    'state_time_integration': OCP_STATE_TIME_INTEGRATION,
    'adjoint_time_integration': OCP_ADJOINT_TIME_INTEGRATION,
    'gradient_calculation': OCP_GRADIENT_CALCULATION,
    #
    'max_iter': OCP_MAX_ITER,
    'norm_gradient_tol': OCP_NORM_GRADIENT_TOL,
    'relative_norm_gradient_tol': OCP_RELATIVE_NORM_GRADIENT_TOL,
    'use_stopping_criterion': OCP_USE_STOPPING_CRITERON,
    #
    'debug': OCP_DEBUG,
    }

class OptimalControlProblem:

    def __init__(
            self,
            system: NonlinearSystem,
            running_cost: callable,
            final_time: float,
            terminal_cost: callable = None,
            initial_control: jnp.ndarray = None,
            options: dict = OCP_OPTIONS,
            ):
        """
        performs setup for optimal control problem

        minimize J(u) = int_0^T running_cost(z(t),u(t)) dt + terminal_cost(z(T))

        subject to the dynamics

        d/dt z = dynamics(z, u), z(0) = initial_state

        described in the class NonlinearSystem

        Args:
            system: instance of NonlinearSystem, with intial state and dynamics
            running_cost: running cost of the optimization problem (non vectorized)
            final_time: final time of the optimization problem
            terminal_cost: (optional) terminal cost of the optimization problem
            initial_control: (optional) initial control guess
            options: (optional) dictionary of options
        """

        # system properties
        self.system = system
        self.dynamics = system.dynamics
        self.initial_state = system.initial_state
        self.nsys = system.nsys
        self.ncontrol = system.ncontrol

        # other fundamentals
        self.running_cost = running_cost
        self.final_time = final_time
        if terminal_cost is None:
            self.terminal_cost = lambda z: 0.0
        else:
            self.terminal_cost = terminal_cost

        # gradient calculation parameters
        if initial_control is not None:
            print(f'\n{style.info}[optimal control] guessing number of time discretization steps based on shape of initial_control\n{style.end}')
            self.nt = initial_control.shape[0]
            self.initial_control = initial_control
        else:
            self.nt = options.get('nt', OCP_OPTIONS['nt'])
            self.initial_control = jnp.zeros((self.nt,self.ncontrol))

        # print(f'norm of initial control = {jnp.linalg.norm(self.initial_control)}')

        self.tt = jnp.linspace(0, self.final_time, self.nt)
        self.state_time_integration = options.get('state_time_integration', OCP_OPTIONS['state_time_integration'])
        self.adjoint_time_integration = options.get('adjoint_time_integration', OCP_OPTIONS['adjoint_time_integration'])
        self.gradient_calculation = options.get('gradient_calculation', OCP_OPTIONS['gradient_calculation'])

        # gradient descent settings
        self.max_iter = options.get('max_iter', OCP_OPTIONS['max_iter'])
        self.norm_gradient_tol = options.get('norm_gradient_tol', OCP_OPTIONS['norm_gradient_tol'])
        self.relative_norm_gradient_tol = options.get('relative_norm_gradient_tol', OCP_OPTIONS['relative_norm_gradient_tol'])
        self.use_stopping_criterion = options.get('use_stopping_criterion', OCP_OPTIONS['use_stopping_criterion'])

        # compute necessary gradients with jax
        self.dynamics_derivative_state = jacfwd(self.dynamics, argnums=0)
        self.dynamics_derivative_control = jacfwd(self.dynamics, argnums=1)
        self.running_cost_derivative_state = grad(self.running_cost, argnums=0)
        self.running_cost_derivative_control = grad(self.running_cost, argnums=1)
        self.terminal_cost_derivative = grad(self.terminal_cost)

        # vmap
        self.vmap_dynamics_derivative_control = vmap(self.dynamics_derivative_control, in_axes=0, out_axes=0,)
        self.vmap_running_cost = vmap(self.running_cost, in_axes=0, out_axes=0,)
        self.vmap_running_cost_derivative_control = vmap(self.running_cost_derivative_control, in_axes=0, out_axes=0,)

        # setup for later use
        self.initial_gradient_norm = None
        self.optimal_control = None
        self.optimal_state = None
        self.optimal_adjoint = None
        self.all_costs = None
        self.all_norm_gradients = None
        self.all_relative_norm_gradients = None

        # store supplied options
        self.options = options

        # debug flag
        self.debug = options.get('debug', OCP_OPTIONS['debug'])

        return

    # control to state map
    def get_state(
            self,
            control,
            ):

        # jax.debug.print(f'{style.warning}norm of control in get_state = {{norm}}{style.end}', norm=jnp.linalg.norm(control))
        # jax.debug.print(f'{style.warning}norm of self.initial_state in get_state = {{norm}}{style.end}', norm=jnp.linalg.norm(self.initial_state))
        
        if self.state_time_integration == 'discrete gradient':
            # the user needs to ensure that these calls are possible
            return discrete_gradient(
                self.system.r,
                self.system.ham_eta,
                self.system.g,
                self.tt,
                self.initial_state,
                control,
                )
        elif self.state_time_integration == 'implicit midpoint':
            return implicit_midpoint(
                self.dynamics,
                self.tt,
                self.initial_state,
                control,
                )
        else:
            raise NotImplementedError('only discrete gradient and implicit midpoint are implemented for state solve')

        
    # adjoint dynamics
    def adjoint_dynamics(
            self,
            adjoint_state,
            adjoint_control
            ):

        state = adjoint_control[:self.nsys]
        control = adjoint_control[self.nsys:]

        return (
            - self.dynamics_derivative_state(state, control).T @ adjoint_state
            - self.running_cost_derivative_state(state, control)
        )

    # state + control to adjoint map
    def get_adjoint(
            self,
            state,
            control
            ):
        """
        solves the adjoint equation

        d/dt p = - D_z dynamics(z,u)^T p - nabla_z running_cost(z,u)
        p(T) = nabla_z terminal_cost(z(T))

        Args:
            state: state trajectory
            control: control trajectory

        Returns:
            adjoint trajectory
        """

        adjoint_control = jnp.concatenate((state, control), axis=1)
        adjoint_final_state = self.terminal_cost_derivative(state[-1,:])
        
        if self.adjoint_time_integration == 'implicit midpoint':
            return implicit_midpoint(
                self.adjoint_dynamics,
                self.tt,
                adjoint_final_state,
                adjoint_control,
                type='backward',
                )
        else:
            raise NotImplementedError('only implicit midpoint is implemented for adjoint solve')


    # cost
    def get_cost(
            self,
            control
            ):

        state = self.get_state(control)

        return (
            jnp.trapezoid(
                y=self.vmap_running_cost(state, control),
                x=self.tt,
                )
            + self.terminal_cost(state[-1,:])
        )

    def get_gradient(
            self,
            control,
            ):

        if self.gradient_calculation == 'jax':
            return grad(self.get_cost)(control)

        else:
            state = self.get_state(control) # shape (nt, nsys)
            adjoint = self.get_adjoint(state, control) # shape (nt, nsys)

            return (
                - self.vmap_running_cost_derivative_control(state, control)
                - jnp.einsum('tab,ta->tb', self.vmap_dynamics_derivative_control(state, control), adjoint)
            )


    def gradient_descent_stopping_criterion(
            self,
            tup: tuple,
            ) -> jnp.ndarray:

        (
            iter,
            control,
            gradient,
            stepsize,
            _,
            _,
            _,
        ) = tup

        # returns true if iteration should continue
        # returns false if iteration should stop

        return jnp.logical_and(
                jnp.logical_and(
                    jnp.less( iter, self.max_iter ),  # i < maxIter
                    jnp.logical_and(
                        jnp.greater( jnp.linalg.norm(gradient), self.norm_gradient_tol ), # norm_gradient > norm_gradient_tol
                        jnp.greater( jnp.linalg.norm(gradient)/self.initial_gradient_norm, self.relative_norm_gradient_tol ), # relative_norm_gradient > relative_norm_gradient_tol
                        )
                    ),
                jnp.logical_and(
                    jnp.logical_not(jnp.allclose(stepsize, 0.0)), # not zero stepsize
                    jnp.logical_not(jnp.isinf(stepsize)), # not inf stepsize
                    )
            )

    def get_barzilai_borwein_stepsize(
            self,
            stepcontrol: int,
            dc: jnp.ndarray, # Delta control
            dg: jnp.ndarray, # Delta gradient
            ) -> jnp.ndarray:

        def big_step():
            return - jnp.einsum('tm,tm->', dc, dc) / jnp.einsum('tm,tm->', dc, dg)

        def small_step():
            return - jnp.einsum('tm,tm->', dc, dg) / jnp.einsum('tm,tm->', dg, dg)

        stepsize = jax.lax.cond(
            stepcontrol % 2 == 0,
            big_step,
            small_step,
            )

        # prevent nans
        stepsize = jax.lax.cond(
            jnp.isnan(stepsize),
            lambda x: 0.0,
            lambda x: x,
            stepsize,
            )

        return stepsize

    def gradient_step(
            self,
            tup: tuple,
            ) -> tuple:
        # performs one step of gradient descent.
        # to be used in jax.lax.while_loop.

        (
            iter,
            previous_control,
            previous_gradient,
            previous_stepsize,
            all_costs,
            all_norm_gradients,
            all_relative_norm_gradients,
        ) = tup

        iter += 1

        control = previous_control + previous_stepsize * previous_gradient
        gradient = self.get_gradient(control)

        # compute step size
        Delta_control = control - previous_control
        Delta_gradient = gradient - previous_gradient

        stepsize = self.get_barzilai_borwein_stepsize(
            iter,
            Delta_control,
            Delta_gradient,
            )

        norm_gradient = jnp.linalg.norm(gradient)
        relative_norm_gradient = norm_gradient/self.initial_gradient_norm

        cost = self.get_cost(control)

        all_costs = all_costs.at[iter].set(cost)
        all_norm_gradients = all_norm_gradients.at[iter].set(norm_gradient)
        all_relative_norm_gradients = all_relative_norm_gradients.at[iter].set(relative_norm_gradient)

        if self.debug:
            jax.debug.print(f'{style.bold}iteration {{iter}}{style.end}', iter=iter)
            jax.debug.print('\tstepsize = {stepsize}', stepsize=stepsize)
            jax.debug.print('\tnorm_gradient = {norm_gradient}', norm_gradient=norm_gradient)
            jax.debug.print('\trelative_norm_gradient = {relative_norm_gradient}', relative_norm_gradient=relative_norm_gradient)
            jax.debug.print('\tcost = {cost}', cost=cost)

        return (
            iter,
            control,
            gradient,
            stepsize,
            all_costs,
            all_norm_gradients,
            all_relative_norm_gradients,
            )

    def gradient_descent(
            self,
            title: str = None,
        ) -> dict:
        """
        computes the optimal control u* using gradient descent.

        Args:
            self: instance of OptimalControlProblem
            title: (optional) title of visualization of cost and gradients, if desired

        Returns:
            dictionary with the optimal control and optimization information
        """

        if self.debug: jax.debug.print(f'\n{style.bold}running gradient descent method with {self.gradient_calculation} gradient calculation {style.end}\n')
        s = timer()

        # setup for lax.while_loop
        iter = 0
        control = self.initial_control
        # if self.debug: jax.debug.print(f'{style.info}initial control norm = {{norm}}{style.end}', norm=jnp.linalg.norm(control))
        gradient = self.get_gradient(control)
        # if self.debug: jax.debug.print(f'{style.info}initial gradient norm = {{norm}}{style.end}\n', norm=jnp.linalg.norm(gradient))
        cost = self.get_cost(control)
        if self.debug: jax.debug.print(f'{style.info}initial cost = {{cost}}{style.end}', cost = cost)

        all_costs = jnp.inf*jnp.ones((self.max_iter,))
        all_norm_gradients = jnp.inf*jnp.ones((self.max_iter,))
        all_relative_norm_gradients = jnp.inf*jnp.ones((self.max_iter,))

        # initial gradient setup
        initial_gradient = gradient
        initial_gradient_norm = jnp.linalg.norm(initial_gradient)
        self.initial_gradient_norm = initial_gradient_norm

        stepsize = 0.5 # first stepsize

        if self.use_stopping_criterion:
            # implementation with jax.lax.while_loop - with stopping criterion
            (
                iter,
                control,
                gradient,
                stepsize,
                all_costs,
                all_norm_gradients,
                all_relative_norm_gradients,
            ) = jax.lax.while_loop(
                self.gradient_descent_stopping_criterion,
                self.gradient_step,
                (
                    iter,
                    control,
                    gradient,
                    stepsize,
                    all_costs,
                    all_norm_gradients,
                    all_relative_norm_gradients,
                    )
                )
        else:
            # implementation with jax.lax.fori_loop with given maximum number of steps - can be differentiated, but a bit risky
            (
                iter,
                control,
                gradient,
                stepsize,
                all_costs,
                all_norm_gradients,
                all_relative_norm_gradients,
            ) = jax.lax.fori_loop(
                0,
                self.max_iter,
                lambda i, x: self.gradient_step(x),
                (
                    iter,
                    control,
                    gradient,
                    stepsize,
                    all_costs,
                    all_norm_gradients,
                    all_relative_norm_gradients,
                    )
                )

        # # crop arrays - does not work with jit, so i removed it
        # all_costs = all_costs[:iter+1]
        # all_norm_gradients = all_norm_gradients[:iter+1]
        # all_relative_norm_gradients = all_relative_norm_gradients[:iter+1]

        # store results in self
        self.optimal_control = control
        self.optimal_state = self.get_state(self.optimal_control)
        self.optimal_adjoint = self.get_adjoint(self.optimal_state, self.optimal_control)
        self.all_costs = all_costs
        self.all_norm_gradients = all_norm_gradients
        self.all_relative_norm_gradients = all_relative_norm_gradients

        e = timer()

        # plot cost and descent
        if title is not None: self.plot_cost_and_descent(title=title)

        return {
            'optimal_control': self.optimal_control,
            'optimal_state': self.optimal_state,
            'optimal_adjoint': self.optimal_adjoint,
            'optimal_cost': all_costs[iter],
            'iterations': iter,
            'all_costs': all_costs,
            'all_norm_gradients': all_norm_gradients,
            'all_relative_norm_gradients': all_relative_norm_gradients,
            'time': e-s,
            }

    def plot_cost_and_descent(
            self,
            title: str = '',
            show_options: bool = True,
            ):
        """
        plots the cost and the relative norm of the gradient
        during the optimization process.

        Args:
            title:

        Returns:

        """
        mpl_settings()

        fig = plt.figure()
        ax_cost = fig.add_subplot(2,1,1)
        ax_norms = fig.add_subplot(2,1,2)

        iterations = jnp.arange(self.all_costs.shape[0])

        ax_cost.semilogy(
            iterations,
            self.all_costs,
            color='tab:blue',
            linewidth=1.5,
            label='cost'
            )

        ax_norms.semilogy(
            iterations,
            self.all_relative_norm_gradients,
            color='tab:green',
            linewidth=1.5,
            label='relative norms of computed gradients'
            )

        ax_cost.set_xlabel('iterations')
        ax_norms.set_xlabel('iterations')

        ax_cost.legend()
        ax_norms.legend()

        fig.suptitle(f'{title}')

        if show_options:
            # add textbox with info
            infostr=''
            for key, value in self.options.items():
                k = str(key)\
                    .replace('gradient', 'grad')\
                    .replace('relative', 'rel')
                    # .replace('calculation', 'calc')\
                    # .replace('integration', 'int')\
                infostr += f'{k:<17} = {value}\n'

            infostr = infostr.replace(' ', '~')
            infostr = infostr.replace('\n', r'\vspace{-7pt}\newline ')
            infostr = r'\ttfamily \noindent \tiny ' + infostr
            props = dict(boxstyle='square', facecolor='white', linewidth=0.7)
            ax_cost.text(1.05, 0.5, infostr, transform=ax_cost.transAxes, fontsize=10,
                    verticalalignment='center', bbox=props)

        fig.tight_layout()
        plt.show()

    def check_gradient(
            self,
            ):
        """
        checks the gradient of optimal control problem to be used
        in gradient descent method.

        the gradient is compared to a finite difference quotient
        of the cost functional in one particular direction.

        the used finite-difference quotient can be of first or second order.
        TODO: second order does not work as expected

        Args:
            self: instance of OptimalControlProblem

        Returns:
            array of values for epsilon and the corresponding (relative) differences
        """

        print(f'running gradient check for {self.gradient_calculation} method, {self.nt} time discretization points')

        seed = 123
        key = random.key(seed)
        control = random.uniform(key, shape=(self.nt,self.ncontrol))

        seed = 999
        key = random.key(seed)
        delta_control = random.uniform(key, shape=(self.nt,self.ncontrol))

        gradient = self.get_gradient(control)
        cost = self.get_cost(control)

        if self.gradient_calculation == 'jax':
            lhs = jnp.einsum('tm,tm->', gradient, delta_control) # usual scalar product suffices
        else:
            lhs = jnp.trapezoid(y=jnp.einsum('ta,ta->t', gradient, delta_control), x=self.tt) # primal-dual gradient needs to be integrated

        print(f'lhs = {lhs:.8e}')

        diff_list = []
        eps_list = [2**(-i) for i in range(24)]
        order_of_approximation = 1 # TODO: 2 is not working as expected
        relative = True

        for eps in eps_list:

            if order_of_approximation == 1:
                rhs = ( self.get_cost(control + eps*delta_control) - cost ) / eps
            elif order_of_approximation == 2:
                rhs = ( self.get_cost(control + eps*delta_control) - self.get_cost(control - eps*delta_control) ) / (2*eps)
            else:
                raise ValueError('order_of_approximation must be 1 or 2')

            print(f'rhs = {rhs:.8e} (eps = {eps:.2e})')
            diff = jnp.abs(lhs - rhs)
            if relative:
                diff = diff/jnp.abs(lhs)

            diff_list += [diff]

        # plot
        eps_list = jnp.array(eps_list)
        diff_list = jnp.array(diff_list)

        import matplotlib.pyplot as plt
        from helpers.other import mpl_settings

        mpl_settings()
        color_index = 0

        labeltext = 'absolute error'
        if relative:
            labeltext = 'relative error'

        plt.loglog(
            eps_list,
            diff_list,
            label=labeltext,
            color=plt.cm.tab20(2*color_index),
        )

        # slope line
        slope = 1
        l = 2
        c = diff_list[l]/eps_list[l]**(slope) # find coefficient to match eps^p to curves
        plt.loglog(
            eps_list, c * eps_list**(slope),
            label=f'$\\varepsilon^{slope}$',
            linestyle='--',
            marker='o',
            markersize=7,
            color=plt.cm.tab20(2*color_index + 1),
            zorder=0,
            )

        plt.xlabel(r'$\varepsilon$')
        if self.gradient_calculation == 'jax':
            lhs_text = f'\\nabla J(u)^T \delta u'
        else:
            lhs_text = r'\int_0^T \nabla J(u)^T \delta u \,\mathrm{d}t'
        rhs_text = r'\frac{J(u + \varepsilon \delta u) - J(u)}{\varepsilon}'
        if order_of_approximation == 2:
            rhs_text = r'\frac{J(u + \varepsilon \delta u) - J(u - \varepsilon \delta u)}{2 \varepsilon}'
        nom_text = f'\\lVert {lhs_text} - {rhs_text} \\rVert'
        denom_text = f'\\lVert {lhs_text} \\rVert'
        ylabeltext = nom_text
        if relative:
            ylabeltext = f'\\frac{{{nom_text}}}{{{denom_text}}}'
        ylabeltext = f'${ylabeltext}$'
        plt.ylabel(ylabeltext)
        plt.legend()
        plt.title(f'gradient check for {self.gradient_calculation} method, {self.nt} time discretization points')
        plt.tight_layout()
        plt.show()

        # return eps_list, diff_list
        return


if __name__ == '__main__':

    # activate double precision and debug flags
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    # import example
    from examples.lti import LTI

    # setup optimal control problem
    alpha = 1e-3
    running_cost = lambda z, u: 1/2 * (z.T @ z + alpha * u.T @ u)
    # terminal_cost = lambda z: 1/2 * z.T @ z
    final_time = 1.0

    #
    A = jnp.array([[1.0,2.0],[3.0,4.0]])

    ocp = OptimalControlProblem(
        system = LTI(A=A),
        running_cost = running_cost,
        # terminal_cost = terminal_cost,
        final_time = final_time,
        options={
            'nt': 10000
            }
        )

    # # check gradient
    ocp.check_gradient()

    # perform gradient descent
    _ = ocp.gradient_descent(title='2D LTI system')



