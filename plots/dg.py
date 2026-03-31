#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file tests the proposed discrete gradient method
#
#

if __name__ == '__main__':

    # jax
    import jax
    import jax.numpy as jnp

    # activate double precision and debug flags
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    # time discretization
    from main.time_discretization import implicit_midpoint, discrete_gradient

    # visualization
    import matplotlib.pyplot as plt
    from helpers.visualization import discrete_gradient_powerbalance
    from helpers.other import mpl_settings
    mpl_settings(
        fontsize=18,
        )

    # examples
    from examples.rigid_body import RigidBody
    from examples.pendulum import Pendulum
    from examples.toda import Toda

    # plants
    pendulum = Pendulum()
    rigid_body = RigidBody()
    toda = Toda()

    # control
    def sine_control(t: float) -> jnp.ndarray:
        return jnp.array([jnp.sin(2*t)])

    sine_control = jax.vmap(sine_control, in_axes=0, out_axes=0)
    zero_control = jax.vmap(lambda t: jnp.zeros((pendulum.ncontrol,)), in_axes=0, out_axes=0)

    # parameters
    T = 10.0 # final time
    nt_discrete_gradient = int(T*100)+1

    # general settings
    base_Delta_t = 1e-3
    num_Delta_t_steps = 9
    Delta_t_array = jnp.array([2**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 3 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(2**ref_order_smaller) # Delta t for reference solution

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * 2**(order) + 1 for order in range(num_Delta_t_steps)]))
    nt_ref = 2**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    Delta_t_ref = T/nt_ref

    print(f'\nnt_ref = {nt_ref}\nDelta_t_ref = {Delta_t_ref:e}')
    print(f'\nnt_array = {nt_array}\nDelta_t_array = {Delta_t_array}\n')
    
    # time grid for reference solution
    tt_ref = jnp.linspace(0,T,nt_ref)

    # setup storage for convergence errors
    all_convergence = {}

    # setup convergence and powerbalance plots
    fig_convergence, ax_convergence = plt.subplots()
    fig_balances, ax_balances = plt.subplots()
    ax_convergence.set_xlabel('step size $\\tau$')
    ax_convergence.set_ylabel(r'$\errorstatenodal$')
    ax_balances.set_xlabel('time $t$')
    ax_balances.set_ylabel(r'$\errorenergy$')
    ax_balances.set_ylim([.5e-18, .5e-3])

    # which systems to test
    tested_systems = ['toda', 'rigid_body', 'pendulum']

    print(f'\n------ setup complete -------\n')

    
    for tested_system in tested_systems:

        print(f'\n- tested system: {tested_system} -\n')

        if tested_system == 'toda':
            system = toda
            control = sine_control
            system_label = r'Example~\ref{ex:toda-lattice}~'
        elif tested_system == 'rigid_body':
            system = rigid_body
            control = sine_control
            system_label = r'Example~\ref{ex:rigid-body}~'
        elif tested_system == 'pendulum':
            system = pendulum
            control = sine_control
            system_label = r'Example~\ref{ex:pendulum}~'
        else:
            raise ValueError('unknown system')

        ham_eta = lambda z: (system.hamiltonian(z), system.eta(z))
        hamiltonian_vmap = jax.vmap(system.hamiltonian)

        # compute reference solution with implicit midpoint method
        uu_ref = control(tt_ref)
        zz_ref = implicit_midpoint(system.dynamics, tt_ref, system.initial_state, uu_ref)

        print(f'reference solution obtained!')

        max_errors = []

        for k in range(num_Delta_t_steps):
            nt = int(nt_array[k])

            tt = jnp.linspace(0,T,nt)
            uu = control(tt)

            zz = discrete_gradient(
                system.r,
                system.ham_eta,
                system.g,
                tt,
                system.initial_state,
                uu,
                )

            # eval reference solution on coarse time gitter
            zz_ref_resampled = zz_ref[::2**(k+ref_order_smaller),:]

            diff = zz_ref_resampled - zz

            # calculate relative error
            error = jnp.linalg.norm(diff, axis=1)#/jnp.linalg.norm(zz_bdf, axis=1) # norms along axis 1, since axis 0 are the time points
            error = error/jnp.max(jnp.linalg.norm(zz_ref, axis=1))

            max_error = jnp.max(error) # Linf error
            max_errors.append(max_error)

            print(f'done for nt={nt}')

        # rename for intuitive use later on
        convergence_errors = max_errors
        all_convergence[tested_system] = convergence_errors
        
        # plot convergence
        ax_convergence.loglog(Delta_t_array, convergence_errors, label=system_label, linewidth=3.0, marker='o', markersize=4)

        ## energy error
        nt_powerbalance = int(T*100) + 1
        tt_powerbalance = jnp.linspace(0,T,nt_powerbalance)
        uu_powerbalance = control(tt_powerbalance)
        zz_ham = discrete_gradient(
            system.r,
            system.ham_eta,
            system.g,
            tt_powerbalance,
            system.initial_state,
            uu_powerbalance,
            return_hamiltonian=True,
            )
            
        _, powerbalance_errors = discrete_gradient_powerbalance(
            system.r,
            system.ham_eta,
            system.g,
            tt_powerbalance,
            zz_ham,
            uu_powerbalance,
            relative=True,
            )
        
        # plot balances
        ax_balances.semilogy(tt_powerbalance[:-1], powerbalance_errors, label=system_label, linewidth=3.0,  alpha=0.6)


    ### save and show plots
    SAVEPATH = './results/figures/dg/dg_all'
    print('\n\n')

    # convergence
    # compute reference
    index = -2
    min_error_at_index = jnp.inf
    for tested_system in tested_systems:
        convergence_error = all_convergence[tested_system][index]
        if convergence_error < min_error_at_index:
            min_error_at_index = convergence_error
    c2 = min_error_at_index/Delta_t_array[-2]**2 / 10 # / 2 to be a bit smaller
    secondorder = c2*Delta_t_array**2
    ax_convergence.loglog(Delta_t_array, secondorder, label='$\\tau^2$', linewidth=3.0, markersize=4, linestyle='--', color='black', zorder=0, alpha=0.4)
    # beautify
    ax_convergence.legend()
    fig_convergence.tight_layout()
    fig_convergence.savefig(f'{SAVEPATH}_convergence.png')
    fig_convergence.savefig(f'{SAVEPATH}_convergence.pgf')
    print(f'figure saved under {SAVEPATH}_convergence (as pgf and png)')
    fig_convergence.suptitle('convergence')
    fig_convergence.tight_layout()
    fig_convergence.show()
    
    # balances
    ax_balances.legend()
    fig_balances.tight_layout()
    fig_balances.savefig(f'{SAVEPATH}_balances.png')
    fig_balances.savefig(f'{SAVEPATH}_balances.pgf')
    print(f'figure saved under {SAVEPATH}_balances (as pgf and png)')
    fig_balances.suptitle('balances')
    fig_balances.tight_layout()
    fig_balances.show()
