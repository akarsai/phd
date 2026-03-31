
#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file tests the performance of the proposed controller.
#
#

if __name__ == '__main__':
    
    SAVEPATH = './results/figures/pc'

    # jax
    import jax
    import jax.numpy as jnp

    # activate double precision and debug flags
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    # time discretization
    from main.time_discretization import implicit_midpoint

    # controller
    from main.controller import Controller

    # visualization
    import matplotlib.pyplot as plt
    from helpers.visualization import plot_controller_trajectory, plot_coupled_trajectory, plot_state_comparison
    from helpers.other import mpl_settings
    mpl_settings(fontsize=18)

    # examples
    from examples.pendulum import Pendulum
    from examples.van_der_pol import VanDerPol

    # set up plants
    pendulum = Pendulum()
    vanderpol = VanDerPol()

    # compute or load uncontrolled plant dynamics

    for tested_system in ['pendulum', 'van_der_pol']:

        print(f'\n\nrunning tests for {tested_system}\n\n')

        # default settings, overwritten for van der pol oscillator
        ocp_cutoff_time = 20.0
        ocp_nt =  100
        simulation_final_time = 10.0

        if tested_system == 'pendulum':
            plant = pendulum
        elif tested_system == 'van_der_pol':
            plant = vanderpol
        else:
            raise NotImplementedError('other systems are not implemented')

        simulation_nt = 500 * int(simulation_final_time/10.0)
        simulation_tt = jnp.linspace(0, simulation_final_time, simulation_nt)

        # compute uncontrolled plant trajectory
        uncontrolled_plant_trajectory = implicit_midpoint(
            plant.dynamics,
            simulation_tt,
            plant.initial_state,
            jnp.zeros((simulation_nt, plant.ncontrol)),
            )

        # plot uncontrolled plant trajectory
        plot_controller_trajectory(
            controller_trajectory = uncontrolled_plant_trajectory,
            tt = simulation_tt,
            axtitle = 'plant',
            variable_name = 'z',
            savepath = f'{SAVEPATH}/{tested_system}_uncontrolled_plant_trajectory',
            )

        # test controller
        controller_initial_state = None
        ocp_initial_control_routine = None
        controller_ylim = None

        if tested_system == 'pendulum':
            options = {
                'ocp_max_iter': 50,
                'pol_xlims': jnp.array([-2*jnp.pi, 2*jnp.pi]),
                'pol_ylims': jnp.array([-2*jnp.pi, 2*jnp.pi]),
                }
            controller_ylim = (-1.89, 1.89)
        elif tested_system == 'van_der_pol':
            options = {
                'pol_degree_x': 15,
                'pol_degree_y': 15,
                'ocp_max_iter': 80,
                }
            controller_ylim = (-0.33, 0.33)
        else:
            raise NotImplementedError('other systems are not implemented')

        controller = Controller(
            plant,
            ocp_initial_control_routine = ocp_initial_control_routine,
            controller_initial_state = controller_initial_state,
            options = options,
            use_policy_iteration = True,
            value_function_savepath = f'{SAVEPATH}/{tested_system}',
            )

        # compute  coupled trajectory
        coupled_trajectory = controller.simulate_coupled_trajectory()

        # plot coupled trajectory
        plot_coupled_trajectory(
            coupled_trajectory = coupled_trajectory,
            tt = controller.simulation_tt,
            controller_ylim = controller_ylim,
            title = f'coupled dynamics for {tested_system}',
            savepath = f'{SAVEPATH}/{tested_system}_coupled_trajectory',
            )


        # compute coupled dynamics with EKF observer gain
        coupled_trajectory_ekf = controller.simulate_coupled_trajectory(use_ekf=True)

        # plot coupled trajectory with EKF observer gain
        plot_coupled_trajectory(
            coupled_trajectory = coupled_trajectory_ekf,
            tt = controller.simulation_tt,
            controller_trajectory_name = '\\zekf',
            controller_ylim = controller_ylim,
            title = f'coupled dynamics for {tested_system} with EKF observer gain',
            savepath = f'{SAVEPATH}/{tested_system}_coupled_trajectory_ekf',
            )

        # plot state decay for uncontrolled plant, coupled plant with our controller, and coupled plant with EKF observer gain
        plot_state_comparison(
            [
                (controller.simulation_tt, uncontrolled_plant_trajectory, 'uncontrolled', 'black'),
                (controller.simulation_tt, coupled_trajectory[:,:controller.plant.nsys], r'controller~\eqref{eq:pc:passive-controller}', plt.cm.tab10(0) ),
                (controller.simulation_tt, coupled_trajectory_ekf[:,:controller.plant.nsys], r'controller~\eqref{eq:pc:ekf-controller}~', plt.cm.tab10(1) ),
                ],
            savepath = f'{SAVEPATH}/{tested_system}_state_decay',
            )

