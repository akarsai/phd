#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements time discretization methods suitable for
# automatic differentiation via jax
#
# currently, the following methods are implemented:
# - implicit midpoint with linear interpolation of the
#   control
# - two custom discrete gradient methods
# - a custom modified petrov-galerkin method
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# custom imports
from helpers.newton import newton_lineax
from helpers.other import style
# for projection method
from scipy.special import roots_legendre
from helpers.energy_based_model import EnergyBasedModel
from helpers.legendre import scaled_legendre, scaled_legendre_on_boundaries
from helpers.gauss import gauss_quadrature_with_values, project_with_gauss

def implicit_midpoint(
        f: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        type = 'forward',
        debug = False,
        ) -> jnp.ndarray:

    """
    uses implicit midpoint method to solve the initial value problem

    z' = f(z,u), z(tt[0]) = z0    (if type == 'forward')

    or

    p' = f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    in the implementation, the control input is linearly interpolated
    to evaluate at midpoints of the time interval.

    :param f: right hand side of ode, f = f(z,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param z0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        z[i,:] = z(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(z0) # system dimension
    nt = len(tt) # number of timepoints
    dt = tt[1] - tt[0] # timestep, assumed to be constant
    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    def F_implicit_midpoint(zj, zjm1, uj12):
        return \
            zj \
            - zjm1 \
            - dt*f( 1/2*(zjm1+zj), uj12)

    solver = newton_lineax(F_implicit_midpoint, debug=debug)

    if type == 'forward':

        z = jnp.zeros((nt,N))
        z = z.at[0,:].set(z0)

        # after that bdf method
        def body( j, var ):
            z, uumid = var

            zjm1 = z[j-1,:]
            uj12 = uumid[j-1,:]

            y = solver(zjm1, zjm1, uj12)
            z = z.at[j,:].set(y)

            if debug: jax.debug.print(f'{style.info}timestep number = {{j}}{style.end}', j=j)
            # jax.debug.print( 'iter = {x}', x = i)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,zjm1,zjm2,zjm3,zjm4,uj))) )

            return z, uumid

        z, _ = jax.lax.fori_loop(1, nt, body, (z,uumid))

        return z

    else: # type == 'backward'

        return implicit_midpoint(f, tt[::-1], z0, uu[::-1,:], type='forward')[::-1,:]
        

def discrete_gradient(
        r: callable,
        ham_eta: callable,
        g: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        debug: bool = False,
        return_hamiltonian: bool = False,
    ) -> jnp.array:
    """
    computes an approximate solution of

    z' = -r(eta(z)) + g(z) u,    z(0) = z_0

    on given timesteps in the time horizon [0,T].
    the system describes passive dynamics, where
    the function r satisfies

    eta(z)^T r(eta(z)) >= 0 for all z

    and eta is the gradient of the hamiltonian function.

    the method uses the gonzalez discrete gradient

    :param r: function r in the system dynamics
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param g: matrix B in the system dynamics
    :param tt: array of timepoints to be used
    :param z0: initial condition
    :param uu: value of control input at timepoints
    :param debug: (optional) debug flag
    :param return_hamiltonian: (optional) if True, also the hamiltonian values are returned
    :return: values of solution at time points [t_0, t_1, ... ]
    """

    nt = tt.shape[0]
    nsys = z0.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant

    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    def get_eta_bar(z, zhat, ham_z):
        # computes eta_bar
        # tries to minimize function calls to ham_eta

        # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
        ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)
        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
        alpha2 = (zhat - z).T @ (zhat - z)

        def true_fun():
            return eta_mid

        def false_fun():
            return eta_mid + alpha1/alpha2 * (zhat - z)

        return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

    def F(zip1, zi, ham_zi, umid):

        return (
            zip1
            - zi
            + Delta_t * r(get_eta_bar(zi, zip1, ham_zi))
            - Delta_t * g(1/2 * (zi + zip1)) @ umid
        )

    solver = newton_lineax(F, debug=debug)

    # set initial condition and hamiltonian array
    z = jnp.zeros((nt,nsys)).at[0,:].set(z0)
    ham_z0, _ = ham_eta(z0)
    ham = jnp.zeros((nt,)).at[0].set(ham_z0)

    # loop
    def body( i, var ):
        z, ham, uumid = var

        zi = z[i,:]
        ham_zi = ham[i]
        umidi = uumid[i,:]

        y = solver(zi, zi, ham_zi, umidi)
        z = z.at[i+1,:].set(y)
        ham_zi, _ = ham_eta(y) # hamiltonian(zkp1)
        ham = ham.at[i+1].set(ham_zi)

        if debug: jax.debug.print('timestep number = {i}', i=i)

        return z, ham, uumid

    z, ham, _ = jax.lax.fori_loop(0, nt-1, body, (z, ham, uumid))

    if return_hamiltonian:
        return z, ham

    return z


def QSR_discrete_gradient(
        f: callable,
        g: callable,
        k: callable,
        ham_eta: callable,
        ell: callable,
        W: callable,
        QSR: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        debug: bool = False,
        return_hamiltonian: bool = False,
    ) -> jnp.array:
    """
    computes an approximate solution of

    z' = f(z) + g(z) u,    z(0) = z_0
    y  = h(z) + k(z) u

    on given timesteps in the time horizon [0,T].
    the system is assumed to describe dissipative dynamics
    with a quadratic supply rate given by

    w(u,y) = y^T Q y + 2 y^T S u + u^T R u.

    the functions eta, ell and W are assumed to satisfy the
    hill-moylan conditions

    eta(z)^T f(z) = h(z)^T Q h(z) - ell(z)^T ell(z)
    1/2 eta(z)^T g(z) = h(z)^T (Q D(z) + S) - ell(z)^T W(z)
    W(z)^T W(z) = R + k(z)^T S + S^T k(z) + k(z)^T Q k(z)

    where eta(z) = \nabla hamiltonian(z) and hamiltonian(z)
    is a storage function of the system.

    in the method, Q D(z) + S is assumed to be invertible.
    hence, h is not needed for the algorithm, as it can be
    recovered using

    h(z) = (Q k(z) + S)^{-T} (1/2 g(z)^T eta(z) + W(z)^T ell(z))

    ---
    the call signature of eta is assumed to be

    ham_eta: z -> hamiltonian(z), eta(z)

    to minimize function calls to hamiltonian in case the
    computation is expensive.
    ---

    the method uses the gonzalez discrete gradient

    :param f: function f in the system dynamics
    :param g: function g in the system dynamics
    :param k: function k in the system dynamics
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param ell: function ell in the hill-moylan conditions
    :param W: function W in the hill-moylan conditions
    :param QSR: tuple of matrices Q, S, R in the supply rate
    :param tt: array of timepoints to be used
    :param z0: initial condition
    :param uu: value of control input at timepoints
    :param debug: (optional) debug flag
    :param return_hamiltonian: (optional) if True, also the hamiltonian values are returned
    :return: values of solution at time points [t_0, t_1, ... ]
    """

    nt = tt.shape[0]
    nsys = z0.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant

    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input
    Q, S, R = QSR

    def get_uncontrolled_discrete_dynamics(z, zhat, ham_z):
        # tries to minimize function calls to ham_eta

        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        def get_eta_bar():
            # computes eta_bar

            # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
            ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)

            alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
            alpha2 = (zhat - z).T @ (zhat - z)

            def true_fun():
                return eta_mid

            def false_fun():
                return eta_mid + alpha1/alpha2 * (zhat - z)

            return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

        eta_bar = get_eta_bar()
        norm_eta_bar = jnp.linalg.norm(eta_bar)
        eta_bar_rescaled = eta_bar.reshape((-1,1)) / norm_eta_bar # has norm 1 and shape (n,1) - without the reshape, this does not work!
        projection = jnp.eye(nsys) - eta_bar_rescaled @ eta_bar_rescaled.T # projects onto span(eta)^\perp
        f_mid = f(1/2 * (z + zhat))
        g_mid = g(1/2 * (z + zhat))
        k_mid = k(1/2 * (z + zhat))
        ell_mid = ell(1/2 * (z + zhat))
        W_mid = W(1/2 * (z + zhat))
        h_mid = jnp.linalg.solve(Q @ k_mid + S, 1/2 * g_mid.T @ eta_bar + W_mid.T @ ell_mid)

        def get_gamma():
            """
            computes gamma according to formula

            gamma = (h_mid^T Q h_mid - ell_mid^T ell_mid)/(eta_bar^T eta_bar)
            """
            def true_fun():
                return 0.0

            def false_fun():
                return (h_mid.T @ Q @ h_mid - ell_mid.T @ ell_mid) / (norm_eta_bar**2)

            return jax.lax.cond(jnp.allclose(norm_eta_bar, 0.0), true_fun, false_fun)

        gamma = get_gamma()

        uncontrolled_discrete_dynamics = (
                gamma * eta_bar
                + projection @ f_mid
            )

        return uncontrolled_discrete_dynamics


    def F(zip1, zi, ham_zi, u_mid):

        g_mid = g(1/2 * (zi + zip1))

        return (
            zip1
            - zi
            - Delta_t * (
                get_uncontrolled_discrete_dynamics(zi, zip1, ham_zi)
                + g_mid @ u_mid
            )
        )

    solver = newton_lineax(F, debug=debug)

    # set initial condition and hamiltonian array
    z = jnp.zeros((nt,nsys)).at[0,:].set( z0 )
    ham_z0, _ = ham_eta(z0)
    ham = jnp.zeros((nt,)).at[0].set(ham_z0)

    # loop
    def body( i, var ):
        z, ham, uumid = var

        zk = z[i,:]
        ham_zk = ham[i]
        umidk = uumid[i,:]

        y = solver(zk, zk, ham_zk, umidk)
        z = z.at[i+1,:].set(y)
        ham_zkp1, _ = ham_eta(y) # hamiltonian(zkp1)
        ham = ham.at[i+1].set(ham_zkp1)

        # jax.debug.print('timestep number = {k}', k=k)
        if debug: jax.debug.print('timestep number = {i}', i=i)

        return z, ham, uumid

    z, ham, _ = jax.lax.fori_loop(0, nt-1, body, (z, ham, uumid))

    if return_hamiltonian:
        return z, ham

    return z


def projection_method(
        ebm: EnergyBasedModel,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        control: callable,
        degree: int,
        num_quad_nodes: int = None,
        num_proj_nodes: int = None,
        debug: bool = False,
        g_manufactured_solution: callable = None,
    ) -> jnp.array:
    """
    computes an approximate solution of
    
    [ nabla_1 ham ]                [    dt z1    ]
    [    dt z2    ]  =  ( J - R )  [ nabla_2 ham ] + B u + g,    z(0) = z_0
    [      0      ]                [     z3      ]
    
    on given timesteps in the time horizon [0,T].

    the method finds an approximate solution ztau such that
    the function ztau satisfies the projection equation and
    is a piecewise polynomial with a specified maximal degree.
    
    note that J and R are not necessarily linear operators.

    :param ebm: energy based model
    :param tt: array of timepoints to be used
    :param z0: initial condition for (z1, z2, z3) that is consistent with the algebraic constraints
    :param control: control function u, callable
    :param degree: degree of piecewise polynomial approximation
    :param num_quad_nodes: number of quadrature nodes used for the quadrature rule for (J-R) eta + B u (optional, default = degree)
    :param num_proj_nodes: number of quadrature nodes used in the projection of eta (optional, default = degree)
    :param debug: debug flag (default False)
    :param g_manufactured_solution: optional function for using a manufactured solution
    :return: solution as a dictionary containing:
                - value of solution at time points [t_0, t_1, ... ]
                # - value of solution at 25 points in each interval [t_0, t_1], [t_1, t_2], ...
                # - value of solution at gauss points in each interval [t_0, t_1], [t_1, t_2], ...
                - array of coefficients for solution of shape (N, (n+1), D)
                    the coefficient for the k-th time interval [t_k, t_{k+1}] is
                    stored in coefflist[k,:,:].
                    evaluation of these coefficients is possible using, e.g.,
                        jnp.einsum('MD,Mt->tD',coefflist[k,:,:],scaled_legendre(n,wanted_points)[0])
                - the degree
                - the number of quadrature nodes for the quadrature rules
                - the number of quadrature nodes for the projection
    """

    # eta^inv is not needed if eta is linear or J and R are state-independent.
    J = ebm.J_vmap
    R = ebm.R_vmap
    B = ebm.B_vmap
    nabla_1_ham = ebm.nabla_1_ham_vmap
    nabla_2_ham = ebm.nabla_2_ham_vmap
    g = g_manufactured_solution
    
    d1, d2, d3 = ebm.dims
    M = tt.shape[0]-1 # number of timepoints
    
    if num_quad_nodes is None:
        num_quad_nodes = degree + 1 # plus one for exact integration of algebraic constraints

    if num_proj_nodes is None:
        num_proj_nodes = degree

    if g is None:
        @jax.profiler.annotate_function
        def g(t):
            return jnp.zeros((t.shape[0],d1+d2+d3))

    # setup variable gauss quadrature for rhs terms -> nqn nodes (nqn = `num quadrature nodes`)
    nqn_gauss_points, nqn_gauss_weights = roots_legendre(num_quad_nodes)
    nqn_gauss_points, nqn_gauss_weights = jnp.array(nqn_gauss_points), jnp.array(nqn_gauss_weights)
    # get legendre values on nqn_gauss_points (for quadratures Q_i)
    psi_at_nqn_gauss_points, psi_prime_at_nqn_gauss_points = scaled_legendre(degree, nqn_gauss_points) # n in first argument is correct here. shape (n+1, nqn)
    phi_at_nqn_gauss_points = psi_at_nqn_gauss_points[:-1,:] # phi[i,j] = phi_i(tj), one degree less since we only test with derivatives for z1 and z2. shape (n, nqn)
    
    # setup gauss quadrature of degree `degree` -> for exact integration of dt_z2 term on the lhs
    k_gauss_points, k_gauss_weights = roots_legendre(degree)
    k_gauss_points, k_gauss_weights = jnp.array(k_gauss_points), jnp.array(k_gauss_weights)
    # get legendre values on nqn_gauss_points (for quadratures Q_i)
    psi_at_k_gauss_points, psi_prime_at_k_gauss_points = scaled_legendre(degree, k_gauss_points) # n in first argument is correct here. shape (n+1, k)
    phi_at_k_gauss_points = psi_at_k_gauss_points[:-1,:] # phi[i,j] = phi_i(tj), one degree less since we only test with derivatives for z1 and z2. shape (n, k)
    
    # setup gauss quadrature for projections -> proj_nodes nodes (npn = `num projection nodes`)
    npn_gauss_points, npn_gauss_weights = roots_legendre(num_proj_nodes)
    npn_gauss_points, npn_gauss_weights = jnp.array(npn_gauss_points), jnp.array(npn_gauss_weights)
    # get legendre values on proj_gauss_points (for scalar product in projection of nabla ham)
    psi_at_npn_gauss_points, psi_prime_at_npn_gauss_points = scaled_legendre(degree, npn_gauss_points) # n in first argument is correct here. shape (n+1, npn)
    phi_at_npn_gauss_points = psi_at_npn_gauss_points[:-1,:] # phi[i,j] = phi_i(tj), one degree less since we only test with derivatives for z1 and z2. shape (n, npn)
    
    # get scaled legendre values on -1 and 1
    minus1, plus1, dt_minus1, dt_plus1 = scaled_legendre_on_boundaries(degree)
    
    # function defining the discretization scheme - solutions of the scheme satisfy F=0
    @jax.profiler.annotate_function
    def F(
        coeffs123: jnp.ndarray,
        left_boundary_value_12: jnp.ndarray,
        k: int,
        ) -> jnp.ndarray:
        """
        this function calculates

        F(coeffs1,coeffs2,coeffs3))

        where zeros of F correspond to coefficients of solutions of the
        scheme for the k-th time interval.

        note: most shifts from [-1,1] to [tk,tkp1] are missing here,
        since they do not change the result of the computation. only
        in the calculation of the derivative partial_t z_h, a scaling
        factor is incorporated.

        :param coeffs123: coefficient vectors c^i_0,...,c^i_n for i=1,2,3, corresponding to z1, z2, z3
        :param left_boundary_value_12: value jnp.hstack(z1(tk), z2(tk)) -- no boundary value for z3 to ensure well posedness of the system
        :param k: timestep number
        :return: F(coeffs)
        """
        
        # extract coefficients for z1, z2, z3
        num_coeffs_12 = (degree+1) * (d1+d2)
        coeffs12 = coeffs123[:num_coeffs_12].reshape((degree+1, d1+d2))
        coeffs1 = coeffs12[:, :d1]
        coeffs2 = coeffs12[:, d1:d1+d2]
        coeffs3 = coeffs123[num_coeffs_12:].reshape((degree, d3))
        # jax.debug.print('coeffs1.shape = {x}', x=coeffs1.shape)
        # jax.debug.print('coeffs2.shape = {x}', x=coeffs2.shape)
        # jax.debug.print('coeffs3.shape = {x}', x=coeffs3.shape)
        # coeffs1[k,d] is the d-th entry of the coefficient belong to p_k for z1, and vice versa for z2 and z3

        tk, tkp1 = tt[k], tt[k+1]

        # find dt_z1 at nqn_gauss_points and dt_z2 at n_gauss_points, d = d1 or d = d2
        dt_z1_nqn = jnp.einsum('Nd,Nt->td', coeffs1, psi_prime_at_nqn_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]
        dt_z2_k = jnp.einsum('Nd,Nt->td', coeffs2, psi_prime_at_k_gauss_points) * 2/(tkp1-tk) # factor for compensating for chain rule in shift: f on [-1,1] -> sqrt(2/(b-a)) f((2t - a - b)/(b-a)) on [a,b]

        # find z1, z2 at npn_gauss_points and z3 at nqn_gauss_points and
        z1_npn = jnp.einsum('Nd,Nt->td', coeffs1, psi_at_npn_gauss_points) # d = d1
        z2_npn = jnp.einsum('Nd,Nt->td', coeffs2, psi_at_npn_gauss_points) # d = d2
        z3_nqn = jnp.einsum('Nd,Nt->td', coeffs3, phi_at_nqn_gauss_points) # d = d3
        
        # find nabla_1_ham at npn_gauss_points # npn since this is essentially also a projection of \nabla_1 H
        nabla_1_ham_npn = nabla_1_ham(z1_npn, z2_npn) # shape (t, d1)
        
        # find projection of nabla_2_ham at nqn_gauss_points
        proj_nabla_2_ham_nqn = project_with_gauss(npn_gauss_weights, phi_at_npn_gauss_points, nabla_2_ham(z1_npn, z2_npn), evaluate_with=phi_at_nqn_gauss_points)

        # find values of (J - R) [ P_{n-1}[eta(z)] ] at nqn_gauss_points
        JmR_nqn = J(dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn) - R(dt_z1_nqn, proj_nabla_2_ham_nqn, z3_nqn)

        # find values of B u at nqn_gauss_points
        shifted_nqn_gauss_points = (tkp1-tk)/2 * nqn_gauss_points + (tk+tkp1)/2
        Bu_nqn = B(control(shifted_nqn_gauss_points))

        # for manufactured solution
        g_nqn = g(shifted_nqn_gauss_points)

        # piece them together
        JmR_nqn = JmR_nqn + Bu_nqn + g_nqn # shape (t,D)

        # left hand side integrals, the ones for z2 are exact
        lhs_prod_1 = jnp.einsum('td,nt->tnd', nabla_1_ham_npn, phi_at_npn_gauss_points) # shape (tpn, n, d1)
        lhs_integrals_1 = gauss_quadrature_with_values(npn_gauss_weights, lhs_prod_1, interval=(tk,tkp1)) # shape (n,d1)
        lhs_prod_2 = jnp.einsum('td,nt->tnd', dt_z2_k, phi_at_k_gauss_points) # shape (tk, n, d1+d2)
        lhs_integrals_2 = gauss_quadrature_with_values(k_gauss_weights, lhs_prod_2, interval=(tk,tkp1)) # shape (n,d2)
        lhs_integrals_12 = jnp.hstack((lhs_integrals_1, lhs_integrals_2))
        
        # right hand side integrals
        JmR_test_12 = jnp.einsum('td,nt->tnd', JmR_nqn[:,:d1+d2], phi_at_nqn_gauss_points) # shape (tqn, n, d1+d2)
        JmR_test_3 = jnp.einsum('td,Nt->tNd', JmR_nqn[:,d1+d2:], phi_at_nqn_gauss_points) # shape (tqn, n+1, d3)
        
        rhs_integrals_12 = gauss_quadrature_with_values(nqn_gauss_weights, JmR_test_12, interval=(tk,tkp1)) # shape (n,d1+d2)
        rhs_integrals_3 = gauss_quadrature_with_values(nqn_gauss_weights, JmR_test_3, interval=(tk,tkp1)) # shape (n+1,d3)

        # ..._integrals contains the integral
        #   ..._integrals[n,d] = int < ... , phi_{nd} >
        # where phi_{nd} is phi_n in the d-th row

        # assemble to dynamics array
        dynamics_12 = lhs_integrals_12 - rhs_integrals_12 # shape (n,d1+d2)
        dynamics_3 = rhs_integrals_3 # shape (n,d1+d2) -- since lhs_integrals_3 == 0
        
        # enforce continuity
        left_boundary_value_12_with_coeffs = jnp.einsum('Nd,N->d', coeffs12, minus1) # only continuity for z1 and z2, since z3 is tested with one degree higher than the rest
        continuity_12 = left_boundary_value_12 - left_boundary_value_12_with_coeffs
        
        # stack and reshape, this is zero if the coefficients are correct
        ret = jnp.hstack((dynamics_12.reshape((-1,)), dynamics_3.reshape((-1,)), continuity_12.reshape((-1,))))
        return ret
    
    num_coeffs_12 = (degree+1)*(d1+d2)
    num_coeffs_3 = degree*d3
    coefflist = jnp.zeros((M, num_coeffs_12+num_coeffs_3))
    # coefflist = jnp.zeros((M,degree+1,d1+d2+d3))

    # setup rootfinder for F
    rootfinder = newton_lineax(f=F, max_iter=10, tol=1e-17, debug=debug) # lineax version
    
    # prepare values of z at timepoints tt
    zz = jnp.zeros((M+1,d1+d2+d3)).at[0,:].set(z0)
    dt_zz_12 = jnp.zeros((M,d1+d2)) # derivatives of z1 and z2 and timepoints tt[1:]

    # initialize left_boundary_value
    left_boundary_value_12 = z0[:d1+d2]

    # initialize coefficient guess - constant in time
    coeff_guess_1 = jnp.ones((degree+1,d1)).at[1:,:].set(0)
    coeff_guess_2 = jnp.ones((degree+1,d2)).at[1:,:].set(0)
    coeff_guess_3 = jnp.ones((degree,d3)).at[1:,:].set(0) # one degree less for z3 variable
    coeff_guess = jnp.hstack((
        coeff_guess_1.reshape((-1,)),
        coeff_guess_2.reshape((-1,)),
        coeff_guess_3.reshape((-1,))
        )) # shape ((degree+1)*d1 + (degree+1)*d2 + degree*d3,)

    @jax.profiler.annotate_function
    def body_fun(k, tup):
        coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12 = tup
        
        if debug: jax.debug.print(f'{style.info}timestep number = {{k}}{style.end}', k=k+1)

        # find root with newton
        root = rootfinder(coeff_guess, left_boundary_value_12=left_boundary_value_12, k=k)
        coeffs12 = root[:num_coeffs_12].reshape((degree + 1, d1 + d2))
        coeffs3 = root[num_coeffs_12:].reshape((degree, d3))
        
        # for values at tt
        zz_12_kp1 = jnp.einsum('ND,N->D', coeffs12, plus1)
        zz_3_kp1 = jnp.einsum('nd,n->d', coeffs3, plus1[:-1]) # use phi bounds for z3
        zz_kp1 = jnp.hstack((zz_12_kp1, zz_3_kp1))
        dt_zz_12_kp1 = jnp.einsum('ND,N->D', coeffs12, dt_plus1) * 2/(tt[k+1] - tt[k])
        zz = zz.at[k+1,:].set(zz_kp1)
        dt_zz_12 = dt_zz_12.at[k+1,:].set(dt_zz_12_kp1)

        # update coefflist and coeff_guess
        coefflist = coefflist.at[k,:].set(root)
        coeff_guess = root

        # update left boundary value for next iteration
        left_boundary_value_12 = zz_kp1[:d1+d2]
        
        return coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12 #, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes

        # print(f'iteration k={k} done')
    init_val = (coefflist, coeff_guess, left_boundary_value_12, zz, dt_zz_12) #, zz_superfine, tt_superfine, zz_quad_nodes, tt_quad_nodes)
    coefflist, _, _, zz, dt_zz_12 = jax.lax.fori_loop(0, M, body_fun, init_val)
    
    return {
        'boundaries': (tt, zz, dt_zz_12), # values of solution at time interval boundaries and derivate values on right boundaries (left side derivative)
        'coefflist': coefflist, # shape (M, (n+1)*d1 + (n+1)*d2 + n*d3 )
        # #
        'degree': degree,
        'num_quad_nodes': num_quad_nodes,
        'num_proj_nodes': num_proj_nodes,
        }


