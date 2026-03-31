#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this script generates plots for the manifold turnpike behavior
#
#

# jax
import jax
# activate double precision and debug flags
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp

# time discretization
from main.time_discretization import implicit_midpoint, discrete_gradient

# optimal control
from main.optimal_control import OptimalControlProblem

# AREs
import control as ct

# visualization
import matplotlib.pyplot as plt

# examples
from examples.lti import LTI
from examples.pendulum import Pendulum
from examples.toda import Toda
from examples.quasilinear_wave import QuasilinearWave

# helpers
from helpers.other import style

# save files
import pickle

def visualize_turnpike(
        kind: str,
        alpha: float = 1e-5,
        final_time: float = 20.0,
        nt: int = 1000,
        max_iter_setup: int = 1000,
        max_iter_solve: int = 20000,
        relative_norm_gradient_tol_solve: float = 1e-6,
        end_penalization: bool = True,
        time_integration: str = 'discrete gradient',
        use_pickle: bool = True,
        figsize: tuple[float, float] = None,
        save: bool = True,
        ):
    
    # setup plant
    if kind == 'lti':
        plant = LTI()
    elif kind == 'pendulum':
        plant = Pendulum()
    # elif kind == 'duffing':
    #     plant = Duffing()
    elif kind == 'toda':
        plant = Toda()
    elif kind == 'quasilinear_wave':
        plant = QuasilinearWave()
    else:
        raise NotImplementedError(f'kind {kind} is not implemented')
    
    # setup time horizon for uncontrolled behavior and plotting
    tt = jnp.linspace(0, final_time, nt)
    
    # to save pickle files and figures
    savepath = f'{SAVEPATH}/{kind}'
    picklepath = f'./results/pickle/{kind}'
    
    
    # setup optimal control problem
    running_cost = lambda z, u: plant.h(z).T @ u + 1/2 * alpha * u.T @ u # this is y^T u + 1/2 * alpha * u^T u
    
    picklename = f'{picklepath}_setup_alpha{alpha}_T{final_time}_nt{nt}_maxiter{max_iter_setup}'
    
    ocp_result = None
    
    if use_pickle:
        try:
            with open(f'{picklename}.pickle','rb') as f:
                ocp_result = pickle.load(f)['ocp_result']
            print(f'ocp_result was loaded')
        except FileNotFoundError:
            pass
    
    if ocp_result is None:
        
        ocp = OptimalControlProblem(
            system = plant,
            running_cost = running_cost,
            final_time = final_time,
            # terminal cost is zero here
            options={
                'nt': nt,
                'max_iter': max_iter_setup, # small number first to get an estimate for z^*(\infty)
                'state_time_integration': time_integration,
                }
            )
    
        # perform gradient descent
        ocp_result = ocp.gradient_descent()
    
        # save file
        with open(f'{picklename}.pickle','wb') as f:
            pickle.dump({'ocp_result':ocp_result},f)
        print(f'ocp_result was written')
        
        
    # extract estimate for final value
    z = ocp_result['optimal_state']
    zfinal = z[-1,:]
    u = ocp_result['optimal_control']
    
    if end_penalization:
        # linearize around zfinal
        linearization_point = zfinal
        A = jax.jacobian(plant.f)(linearization_point)
        B = plant.g(linearization_point)
        C = jax.jacobian(plant.h)(linearization_point)
        
        # compute ARE associated with optimal control
        # the ARE is A^T X + X A + (X B + C^T S)^T R^{−1} (X B + C^T S) + C^T Q C = 0,
        # with S = I, R = 1/2 * alpha * I, Q = 0
        # (but ct.care expects different S, namely C^T S, and different Q, namely C^T Q C)
        X, _, _ = ct.care(A=A, B=B, Q=0*jnp.eye(plant.nsys), R=1/2*alpha*jnp.eye(plant.ncontrol), S=C.T@jnp.eye(plant.ncontrol))
        print(f'\n{style.success}{style.bold}ARE solution computed{style.end}\n')
        
        terminal_cost = lambda w: 1/2 * w.T @ X @ w
        
        picklename = f'{picklepath}_solve_alpha{alpha}_T{final_time}_nt{nt}_maxiter{max_iter_solve}_epsrel{relative_norm_gradient_tol_solve}'
        
        ocp_result = None
        
        if use_pickle:
            try:
                with open(f'{picklename}.pickle','rb') as f:
                    ocp_result = pickle.load(f)['ocp_result']
                print(f'ocp_result was loaded')
            except FileNotFoundError:
                pass
        
        if ocp_result is None:
            
            # repeat ocp with terminal cost and more steps
            ocp = OptimalControlProblem(
                system = plant,
                running_cost = running_cost,
                final_time = final_time,
                terminal_cost = terminal_cost,
                options={
                    'nt': nt,
                    'max_iter': max_iter_solve,
                    'relative_norm_gradient_tol': relative_norm_gradient_tol_solve,
                    'state_time_integration': time_integration,
                    }
                )
                
            # perform gradient descent again
            ocp_result = ocp.gradient_descent()
            
            # save file
            with open(f'{picklename}.pickle','wb') as f:
                pickle.dump({'ocp_result':ocp_result},f)
            print(f'ocp_result was written')
            
        # extract state and control
        z = ocp_result['optimal_state']
        u = ocp_result['optimal_control']
    
    ## simulate uncontrolled behavior
    # solve plant dynamics
    zz = discrete_gradient(
        plant.r,
        plant.ham_eta,
        plant.g,
        tt,
        plant.initial_state,
        0*tt.reshape((-1,plant.ncontrol)),
        )
    
    ## visualize state trajectory
    # setup figure
    fig = plt.figure(figsize=figsize)
    
    if kind == 'pendulum':
        gs = fig.add_gridspec(3, 1)
        
        # extract optimal p and q
        z1_optimal = z[:, 0]
        z2_optimal = z[:, 1]
        
        # extract uncontrolled p and q
        z1_uncontrolled = zz[:,0]
        z2_uncontrolled = zz[:,1]
    
        # norm q subplot bottom-left
        ax_q = fig.add_subplot(gs[0, 0])
        ax_q.plot(tt, z1_optimal, label=r'$z_1(t; u^*)$', linewidth=3.0, )
        ax_q.plot(tt, z1_uncontrolled, label=r'$z_1(t;0)$', linewidth=3.0,  zorder=0, color='tab:blue', linestyle='--', alpha=0.4)
        ax_q.legend(loc='upper right')
        # ax_q.set_xticks([])
        ax_q.tick_params(axis='x', labelbottom=False)
        # ax_q.set_xlabel(r'time $t$')
        # ax_q.set_ylabel(r'$\lVert q^*(t) \rVert$')
        
        # norm p subplot spanning top row (both columns)
        ax_p = fig.add_subplot(gs[1, 0])
        ax_p.plot(tt, z2_optimal, label=r'$z_2(t; u^*)$', linewidth=3.0, )
        ax_p.plot(tt, z2_uncontrolled, label=r'$z_2(t;0)$', linewidth=3.0,  zorder=0, color='tab:blue', linestyle='--', alpha=0.4)
        ax_p.plot(tt, 0*tt, label='turnpike', color='tab:red', linewidth=5.0, zorder=0, alpha=0.4)
        ax_p.legend(loc='upper right')
        # ax_p.set_xticks([])
        ax_p.tick_params(axis='x', labelbottom=False)
        # ax_p.set_xlabel(r'time $t$')
        # ax_p.set_ylabel(r'$\lVert p^*(t) \rVert$')
        
        # control u(t) subplot bottom-right
        ax_u = fig.add_subplot(gs[2, 0])
        ax_u.plot(tt, u[:, 0], label=r'$u^*(t)$', linewidth=3.0, color='tab:green')
        ax_u.set_xlabel(r'time $t$')
        # ax_u.set_ylabel(r'$u^*(t)$')
        ax_u.legend(loc='upper right')
    
    elif kind == 'toda':
        gs = fig.add_gridspec(3, 1)
        
        # extract optimal p and q
        q_optimal = z[:, :plant.number_of_particles]
        p_optimal = z[:, plant.number_of_particles:]
        
        # extract uncontrolled p and q
        q_uncontrolled = zz[:,:plant.number_of_particles]
        p_uncontrolled = zz[:,plant.number_of_particles:]
    
        # norm q subplot bottom-left
        ax_q = fig.add_subplot(gs[0, 0])
        ax_q.plot(tt, jnp.linalg.norm(q_optimal, axis=1), label=r'$\lVert q(t; u^*) \rVert$', linewidth=3.0, )
        ax_q.plot(tt, jnp.linalg.norm(q_uncontrolled, axis=1), label=r'$\lVert q(t;0) \rVert$', linewidth=3.0,  zorder=0, color='tab:blue', linestyle='--', alpha=0.4)
        ax_q.legend(loc='upper right')
        # ax_q.set_xticks([])
        ax_q.tick_params(axis='x', labelbottom=False)
        # ax_q.set_xlabel(r'time $t$')
        # ax_q.set_ylabel(r'$\lVert q^*(t) \rVert$')
        
        # norm p subplot spanning top row (both columns)
        ax_p = fig.add_subplot(gs[1, 0])
        ax_p.plot(tt, jnp.linalg.norm(p_optimal, axis=1), label=r'$\lVert p(t; u^*) \rVert$', linewidth=3.0, )
        ax_p.plot(tt, jnp.linalg.norm(p_uncontrolled, axis=1), label=r'$\lVert p(t;0) \rVert$', linewidth=3.0,  zorder=0, color='tab:blue', linestyle='--', alpha=0.4)
        ax_p.plot(tt, 0*tt, label='turnpike', color='tab:red', linewidth=3.0, zorder=0, alpha=0.4)
        ax_p.legend(loc='upper right')
        # ax_p.set_xticks([])
        ax_p.tick_params(axis='x', labelbottom=False)
        # ax_p.set_xlabel(r'time $t$')
        # ax_p.set_ylabel(r'$\lVert p^*(t) \rVert$')
        
        # control u(t) subplot bottom-right
        ax_u = fig.add_subplot(gs[2, 0])
        ax_u.plot(tt, u[:, 0], label=r'$u^*(t)$', linewidth=3.0,  color='tab:green')
        ax_u.set_xlabel(r'time $t$')
        # ax_u.set_ylabel(r'$u^*(t)$')
        ax_u.legend(loc='upper right')


        

    
    # save figure
    if save:
        fig.align_labels()
        fig.tight_layout()
        savepath += f'_manifold_turnpike'
        if not end_penalization:
            savepath += '_no_end_penalization'
        fig.savefig(savepath + '.pgf') # save as pgf
        fig.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')
    
    # show figure
    fig.tight_layout()  # leave space for suptitle
    plt.show()
    
    
    
    


if __name__ == '__main__':

    SAVEPATH = './results/figures/oc'

    from helpers.other import mpl_settings
    mpl_settings(fontsize=20)
    
    # pendulum
    visualize_turnpike(
        kind='pendulum',
        alpha=1e-4,
        final_time=20.0,
        nt=1000,
        max_iter_setup=2000,
        max_iter_solve=30000,
        relative_norm_gradient_tol_solve=1e-11,
        time_integration='discrete gradient',
        use_pickle=True,
        figsize=(5.5,5),
        save=True,
        )

    # toda lattice
    visualize_turnpike(
        kind='toda',
        alpha=1e-3,
        final_time=20.0,
        nt=1000,
        max_iter_setup=2000,
        max_iter_solve=30000,
        relative_norm_gradient_tol_solve=1e-11,
        time_integration='discrete gradient',
        use_pickle=True,
        figsize=(5.5,5),
        save=True,
        )
    


    
    
