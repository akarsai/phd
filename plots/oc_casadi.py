#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this script generates plots for the manifold turnpike behavior
# using casadi
#
#

import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# this script implements the following optimal control problem:
#
# take x = [z1,z2] and the functions
# E, J(x), R(x), eta(x) and B(x)
# as
# E   = [1 0 ; 0 1 ]
# J   = [0 1 ;-1 0 ]
# R(x)   = [1/4*(4 ||x||^2 + 1)^2  0 ; 0 0]
# eta(x) = [2 z1 ; 1 z2]
# B   = [ 1 0 ]
#
# then the system
#
# E x' = (J - R(x))eta(x) + B u
#    y = B^T eta(x)
#
# forms a pH system.
# we aim to minimize the cost function
#
# C(u) = int_0^T y^T u dt.
#
# we can rewrite the system as
#
# dot z1 = -1/2 (4 z1^2 + 4 z2^2 + 1)^2 z1 + z2 + u
# dot z2 = -2 z1
#     y  =  2 z1
#
# and the cost functional as
#
# C(u) = int_0^T 2 z1 u dt.
#
# we aim to minimize C(u) under the given dynamics
# and the boundary values
#
# x(0) = [2 1]     x(T) = [1 1]
#
# the time horizon is 0 to T,
# where T is sufficiently large.
#
#
# what the code does:
# right hand side with integrator scheme with 'cvodes' solver
#
# quadrature for cost functional with 'quad' option -> how exactly does this work?
#
# for setup of the nonlinear problem:
# - one vector w with symbolic variables
# - two vectors containing the lower and upper bounds for each of the symbolic variable in w
# - J for cost tracking
# - additional constraints (such as continuity and the final value constraint) are stored in an array G containing symbolic variables
#
# nonlinear solver used is 'ipopt'
# - J (the cost tracking term) is minimized
# - w is the variable with respective bounds
# - additional constraints stored in g are passed on to the nonlinear optimization.
#
#


if __name__=='__main__':

    from helpers.other import mpl_settings, style
    mpl_settings(fontsize=20)

    ## setup
    # setup data
    z0 = [2,1]
    zT = [1,1]
    T = 10
    discret_steps = 100*max(round(T/10),1)
    # Uad = [-2,2]
    # Uad = [-cs.inf,cs.inf]
    Uad = [-50,50]
    
    # setup state and control
    z = cs.SX.sym('x',2)
    u = cs.SX.sym('u')

    # setup right-hand side
    f = cs.vertcat(-1/2*((4*z[0]**2+4*z[1]**2+1)**2)*z[0]+z[1]+u,-2*z[0])

    # setup cost functional inner term
    h = 2*z[0]*u

    # setup dae dict
    dae = dict(x=z, p=u, ode=f, quad=h)

    ## create solver instance
    nt = discret_steps
    F = cs.integrator('F', 'idas', dae, 0, T/nt)  # for dae solver

    ## symbolic NLP expression
    # empty NLP
    w = []
    lbw = []
    ubw = []
    G = []
    J = 0

    # initial conditions
    Xk = cs.MX.sym('X0',2)
    w += [Xk]
    lbw += z0
    ubw += z0

    # decision variable for each control interval
    for k in range(1,nt+1):
        # local control
        Uname = 'U'+str(k-1)
        Uk = cs.MX.sym(Uname)
        w += [Uk]
        lbw += [Uad[0]]
        ubw += [Uad[1]]

        # call integrator
        Fk = F(x0=Xk,p=Uk)
        J += Fk['qf'] # increase cost term

        # new local state
        Xname = 'X'+str(k)
        Xk = cs.MX.sym(Xname,2)
        w += [Xk]
        lbw += [-cs.inf, -cs.inf]
        ubw += [cs.inf, cs.inf]

        # continuity constraint
        G += [Fk['xf']-Xk]

    # add final value constraint to G
    G += [Fk['xf'] - zT]

    # the structure of w is as follows:
    #
    # w = [z0[0], z0[1],
    #      u1, z1[0], z1[1],
    #      ...
    #      uend, xend[0], xend[1]]

    ## solve the nonlinear problem
    nlp = dict(f=J,g=cs.vertcat(*G),x=cs.vertcat(*w))
    options = {'ipopt.print_level':0}
    solver = cs.nlpsol('S','ipopt',nlp,options)
    result = solver(lbx=lbw,ubx=ubw,x0=0,lbg=0,ubg=0)
    print(f'\n{style.success}{style.bold}solver complete{style.end}\n')

    # prepare solution for return
    wopt = result['x'].toarray()

    # refactor w into parts
    z1opt = [z0[0]]
    z2opt = [z0[1]]
    uopt = []
    # the first two elements are skipped, they belong to the initial condition
    for i,l in enumerate(wopt[2:]):
        if i%3==0:
            uopt += [l.item()]
        elif i%3==1:
            z1opt += [l.item()]
        elif i%3==2:
            z2opt += [l.item()]
    
    ## compute uncontrolled trajectory as comparison
    # Uncontrolled dynamics (u = 0)
    f_uncontrolled = cs.vertcat(
        -1/2*((4*z[0]**2 + 4*z[1]**2 + 1)**2)*z[0] + z[1],
        -2*z[0]
    )
    
    # Setup DAE dictionary without control
    dae = dict(x=z, ode=f_uncontrolled)
    
    # Create integrator
    dt = T / nt
    F = cs.integrator('F', 'cvodes', dae, 0, dt)
    
    # Simulate forward in time
    z_uncontrolled = np.zeros((nt + 1, 2))
    z_uncontrolled[0, :] = z0
    
    z_current = z0
    for k in range(nt):
        Fk = F(x0=z_current)
        z_current = Fk['xf'].full().flatten()
        z_uncontrolled[k+1, :] = z_current

    
    ## visualization
    nt = len(z1opt)-1
    time = np.linspace(0,T,nt+1)

    fig = plt.figure(figsize=(5.5,5))
    gs = fig.add_gridspec(3, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, z1opt, label=r'$z_1(t; u^*)$', linewidth=3.0,)
    ax.plot(time, z_uncontrolled[:,0], label=r'$z_1(t; 0)$', color='tab:blue', linewidth=3.0, alpha=0.4, linestyle='--', zorder=0)
    ax.plot(time, [0 for z in z1opt], label='turnpike', color='tab:red', linewidth=5.0, zorder=0, alpha=0.4)
    ax.legend(loc='upper center')
    ax.tick_params(axis='x', labelbottom=False)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time, z2opt, label='$z_2(t; u^*)$', linewidth=3.0,)
    ax.plot(time, z_uncontrolled[:,1], label='$z_2(t; 0)$', color='tab:blue', linewidth=3.0, alpha=0.6, linestyle='--', zorder=0)
    ax.legend(loc='upper center')
    ax.tick_params(axis='x', labelbottom=False)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(time[1:], uopt, color='tab:green', label=r'$u^*(t)$', linewidth=3.0,)
    ax.legend()
    # ax.set_ylabel('$u^*(t)$')
    ax.set_xlabel('time $t$')
    
    # save figure
    fig.align_labels()
    fig.tight_layout()
    savepath = './results/figures/oc/academic_manifold_turnpike'
    fig.savefig(savepath + '.pgf') # save as pgf
    fig.savefig(savepath + '.png') # save as png
    print(f'figure saved under savepath {savepath} (as pgf and png)')
    
    # show figure
    fig.suptitle(r'\tiny academic example')
    fig.tight_layout()
    plt.show()
    