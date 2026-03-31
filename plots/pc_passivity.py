#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file verifies the passivity of the proposed controller
# on two examples
#
#

if __name__ == '__main__':

    SAVEPATH = './results/figures/pc'

    # jax
    import jax
    import jax.numpy as jnp

    # activate double precision and debug flags
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_debug_nans", True)

    # controller
    from main.controller import Controller

    # time discretization
    from main.time_discretization import implicit_midpoint, discrete_gradient

    # visualization
    import matplotlib.pyplot as plt
    from helpers.visualization import discrete_gradient_powerbalance, visualize_hamiltonian, visualize_errors, plot_coupled_trajectory
    from helpers.other import mpl_settings
    mpl_settings(
        fontsize=18,
        )

    # examples
    from examples.lti import LTI
    from examples.pendulum import Pendulum
    from examples.van_der_pol import VanDerPol

    # plants
    lti = LTI()
    pendulum = Pendulum()
    van_der_pol = VanDerPol()

    # control
    def sine_control(t: float) -> jnp.ndarray:
        return jnp.array([jnp.sin(t)])

    sine_control = jax.vmap(sine_control, in_axes=0, out_axes=0)
    zero_control = jax.vmap(lambda t: jnp.zeros((pendulum.ncontrol,)), in_axes=0, out_axes=0)

    nt_discrete_gradient = 500

    ###  test controller
    visualize_hamiltonian_tuplist = []
    visualize_errors_tuplist = []

    # for tested_system in ['pendulum', 'van_der_pol']:
    for tested_system in ['pendulum', 'van_der_pol']:

        if tested_system == 'pendulum':
            plant = pendulum
            options = {
                'ocp_max_iter': 50,
                # 'ocp_debug': False, # True,
                'simulation_nt': nt_discrete_gradient,
                }
            exampleref = r'ex:pendulum'
        elif tested_system == 'van_der_pol':
            plant = van_der_pol
            options = {
                'pol_degree_x': 15,
                'pol_degree_y': 15,
                'ocp_max_iter': 80,
                'simulation_nt': nt_discrete_gradient,
                }
            exampleref = r'ex:van_der_pol'
        else:
            raise ValueError(f'tested_system {tested_system} not supported')

        controller = Controller(
            plant,
            options = options,
            )

        # test zero control
        control = zero_control
        label = r'0'
        controller.controller_initial_state = jnp.ones((plant.nsys,)) # change initial condition

        # compute controller trajectory
        controller_trajectory_and_hamiltonian = controller.simulate_controller_trajectory(
            control=control,
            # title=f'controller dynamics with zero control ({tested_system})',
            )

        # compute hamiltonian and power balance
        ham, errors = discrete_gradient_powerbalance(
            controller.controller_r,
            controller.controller_ham_eta,
            controller.plant.g,
            controller.simulation_tt,
            controller_trajectory_and_hamiltonian,
            control(controller.simulation_tt),
            relative=True,
            )

        visualize_hamiltonian_tuplist.append(
            (controller.simulation_tt, ham, rf'Example~\ref{{{exampleref}}}')
            )
        
        visualize_errors_tuplist.append(
            (controller.simulation_tt, errors, rf'Example~\ref{{{exampleref}}}')
            )

    # plot hamiltonian
    visualize_hamiltonian(
        visualize_hamiltonian_tuplist,
        title=f'evolution of hamiltonian for controller dynamics',
        ylabeltext=r'$\hamc(\zc(t))$',
        axis_type='semilogy',
        savepath=f'{SAVEPATH}/controller_hamiltonian_zero_control',
        )
    
    # plot power balance error
    visualize_errors(
        visualize_errors_tuplist,
        title=f'relative error in power balance for controller dynamics ({tested_system})',
        savepath=f'{SAVEPATH}/controller_powerbalance_zero_control',
        )
