#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file creates plots for the introduction of the thesis
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
    from helpers.other import mpl_settings, dprint
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
    
    # only consider toda lattice
    system = toda
    
    # extract hamiltonian
    ham_eta = system.ham_eta
    hamiltonian_vmap = jax.vmap(lambda z: system.hamiltonian(jnp.zeros_like(z), z), in_axes=0,)

    # control
    def sine_control(t: float) -> jnp.ndarray:
        return jnp.array([jnp.sin(t)])

    sine_control = jax.vmap(sine_control, in_axes=0, out_axes=0)
    zero_control = jax.vmap(lambda t: jnp.zeros((1,)), in_axes=0, out_axes=0)
    control = zero_control
    
    # parameters
    T = 10.0 # final time
    
    # general settings
    base_Delta_t = 5e-2
    num_Delta_t_steps = 4
    basis = 2
    Delta_t_array = jnp.array([basis**i * base_Delta_t for i in range(num_Delta_t_steps)])
    ref_order_smaller = 2 # by which order of magnitude should the reference solution be smaller than the smallest tested Delta t?
    ref_Delta_t = base_Delta_t/(basis**ref_order_smaller) # Delta t for reference solution

    # convert Delta_t values to nt values
    groß_n = (T/Delta_t_array[-1]).astype(int)
    nt_array = jnp.flip(jnp.array([groß_n * basis**(order) + 1 for order in range(num_Delta_t_steps)]))
    # nt_ref = 10**(ref_order_smaller+num_Delta_t_steps-1) * groß_n + 1
    
    dprint(nt_array)
    # dprint(nt_ref)

    # convert back to get "real" Delta_t corresponding to the nt_spp values
    Delta_t_array = T/nt_array
    # Delta_t_ref = T/nt_ref
    
    # setup plots
    fig_im, ax_im = plt.subplots()
    fig_dg, ax_dg = plt.subplots()
    for ax in [ax_im, ax_dg]:
        ax.set_xlabel('time $t$')
        ax.set_ylabel('$\\mathcal{H}(z^\\tau(t))$')
    
    
    # colors
    cmap_im = plt.get_cmap('OrRd')
    cmap_dg = plt.get_cmap('GnBu')
    colors_im = [cmap_im(x/(num_Delta_t_steps+1)) for x in range(1,num_Delta_t_steps+1)]
    colors_dg = [cmap_dg(x/(num_Delta_t_steps+1)) for x in range(1,num_Delta_t_steps+1)]

    for k in range(num_Delta_t_steps):
        nt = int(nt_array[k])

        # set up array of timepoints
        tt = jnp.linspace(0,T,nt)
        
        # compute control
        uu = control(tt)

        # compute solution with implicit midpoint method
        zz_im = implicit_midpoint(
            system.dynamics,
            tt,
            system.initial_state,
            uu,
            )
        
        # compute solution with discrete gradient method
        zz_dg = discrete_gradient(
            system.r,
            system.ham_eta,
            system.g,
            tt,
            system.initial_state,
            uu,
            )
        
        # compute associated hamiltonians
        hh_im = hamiltonian_vmap(zz_im)
        hh_dg = hamiltonian_vmap(zz_dg)
        
        # plot
        ax_im.plot(tt, hh_im, color=colors_im[k], label=rf'$\tau = {Delta_t_array[k]:.2f}$')
        ax_dg.plot(tt, hh_dg, color=colors_dg[k], label=rf'$\tau = {Delta_t_array[k]:.2f}$')

    ### save and show plots
    SAVEPATH = './results/figures/intro'
    print('\n\n')

    # legends + tight_layout
    ax_im.legend()
    ax_dg.legend()
    fig_im.tight_layout()
    fig_dg.tight_layout()

    # saving
    fig_im.savefig(f'{SAVEPATH}/implicit_midpoint_energy.png')
    fig_im.savefig(f'{SAVEPATH}/implicit_midpoint_energy.pgf')
    fig_dg.savefig(f'{SAVEPATH}/discrete_gradient_energy.png')
    fig_dg.savefig(f'{SAVEPATH}/discrete_gradient_energy.pgf')

    # show
    fig_im.suptitle('implicit midpoint')
    fig_im.tight_layout()
    fig_im.show()
    fig_dg.suptitle('discrete gradient')
    fig_dg.tight_layout()
    fig_dg.show()
