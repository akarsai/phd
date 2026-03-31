#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file is used to generate the visualizations of
# the state of the doubly nonlinear paraboliv model
#
#



if __name__ == '__main__':
    
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    
    from timeit import default_timer as timer
    from itertools import product
    
    from main.time_discretization import projection_method
    from examples.doubly_nonlinear_parabolic import DoublyNonlinearParabolic, DoublyNonlinearParabolicReducedOrder
    
    import pickle
    import matplotlib.pyplot as plt
    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # set savepath
    SAVEPATH = './results'
    
    # simulation settings
    T = 0.1
    nt = 501
    tt = jnp.linspace(0, T, nt)
    degree = 4
    num_quad_nodes = 2*degree # degree leads to difficulties in the newton solver
    num_proj_nodes = 2*degree
    
    # what parameters to test
    plist = [1.5, 2]
    qlist = [1.5, 2, 3]
    nx = 50
    
    for (p,q) in product(plist, qlist):
        
        # set paths
        savepath_fom = f'{SAVEPATH}/figures/mpg/doubly_nonlinear_parabolic_p{p}_q{q}_nx{nx}'
        picklepath = f'{SAVEPATH}/pickle/doubly_nonlinear_parabolic_p{p}_q{q}_nx{nx}'
        
        # setup models
        ebm_fom = DoublyNonlinearParabolic(p=p, q=q, nx=nx)
        ebm_rom = DoublyNonlinearParabolicReducedOrder(p=p, q=q, nx=nx, picklepath=picklepath) # picklepath to load precomputed fom result
        
        # update paths
        savepath_rom = f'{savepath_fom}_rom'
        picklename_fom = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
        picklename_rom = f'{picklepath}_rom_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}'
        
        
        # set up hamiltonian plots
        fig_ham, ax_ham = plt.subplots()
    
        # run fom and rom simulations
        for (ebm, picklename, savepath) in [(ebm_fom, picklename_fom, savepath_fom), (ebm_rom, picklename_rom, savepath_rom)]:
            
            # get default control and initial condition
            control = ebm.default_control
            z0 = ebm.initial_condition
            
            # run simulation with projection method
            try: # try to skip also the evaluation
                with open(f'{picklename}.pickle','rb') as f:
                    proj_solution = pickle.load(f)['proj_solution']
                print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\tresult was loaded')
        
            except FileNotFoundError: # evaluation was not done before
                s = timer()
                proj_solution = projection_method(
                    ebm=ebm,
                    tt=tt,
                    z0=z0,
                    control=control,
                    degree=degree,
                    num_quad_nodes=num_quad_nodes,
                    num_proj_nodes=num_proj_nodes,
                    debug=False,
                    )
                e = timer()
                print(f'({degree = }, {num_quad_nodes = }, {num_proj_nodes = }, {nt = })\n\ttook {e-s:.2f} seconds')
        
                # save file
                with open(f'{picklename}.pickle','wb') as f:
                    pickle.dump({'proj_solution': proj_solution},f)
                print(f'\tresult was written')
            
            zz_proj = proj_solution['boundaries'][1]
        
            zz_proj_1 = zz_proj[:, :ebm.dims[0]]
            zz_proj_2 = zz_proj[:, ebm.dims[0]:ebm.dims[0]+ebm.dims[1]]
            label = r'$\mathcal{H}^h$'
            if ebm.is_rom:
                label = r'$\tilde{\mathcal{H}}$'
            ax_ham.plot(tt, ebm.hamiltonian_vmap(zz_proj_1, zz_proj_2), label=label, linewidth=3.0)
            
            # visualize solution
            ebm.visualize_solution(
                zz=zz_proj,
                tt=tt,
                vmin=-1,
                vmax=1,
                colorbarticks=[-1,0,1],
                savepath=savepath+'_state',
                )
    
        # plot hamiltonians
        ax_ham.set_xlabel('time $t$')
        ax_ham.set_ylabel('energy')
        ax_ham.legend()
        fig_ham.tight_layout()
        fig_ham.savefig(savepath_fom+'_hamiltonian.pgf')
        fig_ham.savefig(savepath_fom+'_hamiltonian.png')
        print(f'figure saved under savepath {savepath_fom}_hamiltonian (as pgf and png)')
        fig_ham.suptitle(f'Energies for doubly nonlinear parabolic model (${p=}, {q=}$)')
        fig_ham.tight_layout()
        fig_ham.show()