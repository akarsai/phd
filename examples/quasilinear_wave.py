#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a quasilinear wave equation.
#
#


import jax
import jax.numpy as jnp


# plotting
import matplotlib.pyplot as plt

# file saving
import pickle

# timing
from timeit import default_timer as timer

# energy based model
from helpers.energy_based_model import EnergyBasedModel_LinearJR, EnergyBasedModel

# discretization
from main.space_discretization import AnsatzSpace1D
from main.time_discretization import projection_method, implicit_midpoint

# sparse computations
from jax.scipy.sparse.linalg import bicgstab

# helpers
from helpers.other import dprint


class QuasilinearWave(EnergyBasedModel):
    """
    implements
    
        dt rho + dx v = 0
        dt v + dx p(rho) = - gamma F(v) + nu dx^2 v
    
    on Omega = [0,L] with boundary conditions
    
        p(rho(t,0)) - nu dx v(t,0) = u_1
        p(rho(t,1)) - nu dx v(t,1) = u_2
        
    with
        
        p(rho) = rho + rho^3
        F(v) = |v|^{s-2} v
    
    we consider u_1 = TODO and u_2 = TODO
    """
    
    def __init__(
            self,
            mesh_settings: dict = None,
            s: int | float = 1.5,
            gamma: float = 0.1,
            nu: int = 0,
            eps_F: float = 1e-307, # smallest number that can be represented, check sys.float_info.min. this is to avoid jax compilation issues!
            nx: int = 25,
            ):
        
        # save parameters
        self.s = s
        self.gamma = gamma
        self.nu = nu
        self.eps_F = eps_F
        
        # define ansatz space for space discretization
        if mesh_settings is None:
            mesh_settings = {
                'L': 1.0,
                }
        
        mesh_settings['n'] = nx
        
        self.space = AnsatzSpace1D(
            mesh_settings=mesh_settings,
            inner_product='H1',
            )
        
        super().__init__()
        
        # set PDE flag
        self.was_pde = True
        
        # set two component flag for error calculatio
        self.has_two_components = True
        
        # set dimensions
        self.dims = (0, 2*self.space.dim, 0) # 0 for z1 and z3 dims, rho + v are both H1 so i need 2*dims
        
        # initial condition - fits default manufactured solution
        self.rho_init = lambda x: jnp.cos(4*jnp.pi*x/self.space.mesh.L)
        # self.rho_init = lambda x: 1 + 1/2 * jnp.sin(jnp.pi*x/self.space.mesh.L)
        self.rho_init_coeffs = self.space.get_projection_coeffs(self.rho_init(self.space.mapped_quad_nodes), inner_product='L2')
        self.v_init = lambda x: jnp.cos(4*jnp.pi*x/self.space.mesh.L)
        # self.v_init = lambda x: (4*x/self.space.mesh.L - 2)**3
        self.v_init_coeffs = self.space.get_projection_coeffs(self.v_init(self.space.mapped_quad_nodes), inner_product='L2')
        self.initial_condition = jnp.hstack((self.rho_init_coeffs, self.v_init_coeffs))
        
        # p(rho_init(0)) - nu*dx_v_init(0) = 2 - 0 = 2
        
        
    def J(self, dt_z1, h2, z3):
        """
        Compute the nonlinear operator:
        <j(p(rho), v), φ> = ∫_Ω - ∇v·φ + p(rho)·∇φ_i dx
        
        for v = h2
        """
        
        prho = h2[:self.space.dim]
        v = h2[self.space.dim:]
        
        # apply inverse mass matrix to preserve structure
        M2inv_prho, _ = bicgstab(self.space.l2_mass_matrix, prho,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        M2inv_v, _ = bicgstab(self.space.l2_mass_matrix, v,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        prho_dx_test = self.space.l2_mixed_matrix.T @ M2inv_prho # transpose is important here
        dx_v_test = - self.space.l2_mixed_matrix @ M2inv_v # no test here, minus sign for correct structure
        
        # multiply with inverse mass matrices to preserve structure
        M2inv_prho_dx_test, _ = bicgstab(self.space.l2_mass_matrix, prho_dx_test,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        M2inv_dx_v_test, _ = bicgstab(self.space.l2_mass_matrix, dx_v_test,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        return jnp.hstack((M2inv_dx_v_test, M2inv_prho_dx_test)) # order is flipped
        
    def R(self, dt_z1, h2, z3):
        """
        Compute the nonlinear operator:
        <r(v), φ> = ∫_Ω   gamma |v|^{s-2} v · φ  +  nu ∇v · ∇φ  dx
        """
        
        v = h2[self.space.dim:]
        
        # apply inverse mass matrix to preserve structure
        M2inv_v, _ = bicgstab(self.space.l2_mass_matrix, v,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        # evaluate v and grad v at quadrature points
        v_quad, _ = self.space.eval_coeffs_quad(M2inv_v)
        # shapes: v_quad (num_elements, num_quad_nodes)
        #         grad_v_quad (num_elements, num_quad_nodes)
        
        # compute ∫_Ω gamma |v|^{s-2} v · φ dx
        v_norm = jnp.abs(v_quad)
        r1_integrand = jnp.einsum('eq,beq->beq', (v_norm + self.eps_F)**(self.s - 2.0) * v_quad, self.space.gbf_quad)
        # shape: (dim, num_elements, num_quad_nodes)
        r1 = self.gamma * jnp.sum(self.space.quadrature_with_values_physical(r1_integrand), axis=1)
        # shape: (dim,)
        
        # compute ∫_Ω  nu ∇v · ∇φ dx
        r2 = self.nu * self.space.l2_stiffness_matrix @ M2inv_v
        # shape: (dim,)
        
        # apply inverse mass matrix to preserve structure
        M2inv_r, _ = bicgstab(self.space.l2_mass_matrix, r1 + r2,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        return jnp.hstack((jnp.zeros_like(M2inv_r), M2inv_r))
    
    def hamiltonian(self, z1, z2):
        """
        Energy functional:
        H(z) = ∫_Ω    1/2 v^2 + 1/4 rho^4 + 1/2 rho^2    dx
        
        Args:
            z1: not used (empty)
            z2: coefficients for z=(rho,v)
            
        Returns:
            scalar energy value
        """
        # evaluate z at quadrature points
        rho = z2[:self.space.dim]
        v = z2[self.space.dim:]
        rho_quad, _ = self.space.eval_coeffs_quad(rho)
        v_quad, _ = self.space.eval_coeffs_quad(v)
        # shape: (num_elements, num_quad_nodes)
        
        # compute 1/2 v^2 + 1/2 rho^2 + 1/4 rho^4
        integrand = 1/2 * v_quad**2 + 1/4 * rho_quad**4 + 1/2 * rho_quad**2
        # shape: (num_elements, num_quad_nodes)
        
        # integrate using quadrature
        energy = jnp.sum(self.space.quadrature_with_values_physical(integrand))
        
        return energy
     
    def B(self, u):
        """
        Control input operator for 1D with Neumann boundary conditions:
        <G u(t), φ> := ∫_Ω u_1(t) φ dx + u_2(L) φ(L) - u_2(0) φ(0)
        
        The boundary terms come from integration by parts in the weak formulation.
        
        Args:
            u: control input, shape (..., 2)
               u[..., 0] = u(0) (left boundary flux control)
               u[..., 1] = u(L) (right boundary flux control)
            
        Returns:
            B_u: vector of shape (..., 2*dim)
        """
        # Single control input
        u_left = u[0] # left boundary control
        u_right = u[1] # right boundary control
        
        # Boundary contribution from weak formulation
        # In 1D: u_2(L) * φ_i(L) - u_2(0) * φ_i(0)
        boundary_contribution = jnp.zeros(self.space.dim)
        
        # Left boundary (x=0): subtract u_2(0) because of outward normal direction
        # For left boundary, outward normal is -1, so we get -u_2(0) * φ_i(0)
        boundary_contribution = boundary_contribution.at[0].set(u_left)
        
        # Right boundary (x=L): add u_2(L) because of outward normal direction
        # For right boundary, outward normal is +1, so we get +u_2(L) * φ_i(L)
        boundary_contribution = boundary_contribution.at[-1].set(-u_right)
        
        b = boundary_contribution
        
        # apply inverse mass matrix to preserve structure
        M2inv_b, _ = bicgstab(self.space.l2_mass_matrix, b,
                            tol=1e-14, atol=1e-14, maxiter=1000)
        
        return jnp.hstack((jnp.zeros_like(M2inv_b), M2inv_b))
    
    def default_control(self, t):
        return 2 - jnp.stack((jnp.sin(t), jnp.sin(t))).T # matches initial condition

    def get_manufactured_solution(self):
        
        def rho_spatial(x):
            return jnp.cos(4*jnp.pi*x/self.space.mesh.L)
            
        timescale = 40*jnp.pi
        
        def rho_manufactured(x, t):
            return jnp.cos(timescale * t) * rho_spatial(x)
        
        def dt_rho_manufactured(x, t):
            return - timescale * jnp.sin(timescale * t) * rho_spatial(x)
        
        def v_spatial(x):
            return jnp.cos(4*jnp.pi*x/self.space.mesh.L)
            
        def v_manufactured(x, t):
            return jnp.cos(timescale * t) * v_spatial(x)
        
        def dt_v_manufactured(x, t):
            return - timescale * jnp.sin(timescale * t) * v_spatial(x)
        
        rho_manufactured_quad = jax.vmap(lambda t: rho_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        dt_rho_manufactured_quad = jax.vmap(lambda t: dt_rho_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        v_manufactured_quad = jax.vmap(lambda t: v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        dt_v_manufactured_quad = jax.vmap(lambda t: dt_v_manufactured(self.space.mapped_quad_nodes, t), in_axes=0)
        projection_vmap = jax.vmap(lambda x: self.space.get_projection_coeffs(x, inner_product='L2'), in_axes=0)
        rho = lambda t: projection_vmap(rho_manufactured_quad(t))
        dt_rho = lambda t: projection_vmap(dt_rho_manufactured_quad(t))
        v = lambda t: projection_vmap(v_manufactured_quad(t))
        dt_v = lambda t: projection_vmap(dt_v_manufactured_quad(t))
        z2 = lambda t: jnp.hstack((rho(t), v(t)))
        dt_z2 = lambda t: jnp.hstack((dt_rho(t), dt_v(t)))
        
        # empty array callable
        _ = lambda t: jnp.zeros((t.shape[0], 0)) # empty array
        
        # z1 and z3 are empty array
        z1 = z3 = dt_z1 = _
        
        # build initial condition
        t0 = jnp.zeros((1,))
        z1_0 = z1(t0)[0,:]
        z2_0 = z2(t0)[0,:]
        z3_0 = z3(t0)[0,:]
        z0 = jnp.hstack((z1_0, z2_0, z3_0))
        
        # build zero control input
        u = jnp.zeros((self.space.dim+2,))
        
        # compute g for g_manufactured solution - the latter is a constant map since v_manufactured is constant
        h1 = lambda t: self.nabla_1_ham_vmap(z1(t), z2(t))
        h2 = lambda t: self.nabla_2_ham_vmap(z1(t), z2(t))
        rhs = lambda t: \
            self.J_vmap(dt_z1(t), h2(t), z3(t)) \
            - self.R_vmap(dt_z1(t), h2(t), z3(t)) \
            + self.B(u) # u is constant, no B_vmap needed. otherwise: self.B_vmap(u(t))
        lhs = lambda t: jnp.hstack((h1(t), dt_z2(t), jnp.zeros_like(z3(t))))
        
        manufactured_solution = lambda t: jnp.hstack((z1(t), z2(t), z3(t)))
        control_manufactured_solution = lambda t: jnp.tile(u, (t.shape[0], 1)) # repeats boundary input entries as often as needed, shape (t.shape[0], ncontrol)
        g_manufactured_solution = lambda t: lhs(t) - rhs(t)
        
        return z0, manufactured_solution, control_manufactured_solution, g_manufactured_solution
    
    def visualize_solution(
        self,
        tt: jnp.ndarray,
        zz: jnp.ndarray,
        vmin: float = None,
        vmax: float = None,
        colorbarticks: list = None,
        title: str = None,
        savepath: str = None,
        interpolation: str = 'gaussian',
        ):
        """
        Visualize the solution at different time steps using imshow.
        
        Parameters
        ----------
        interpolation : str, optional
            Interpolation method for imshow. Options include:
            'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', etc.
            Default is 'bilinear'.
        """
        fig, (ax_rho, ax_v) = plt.subplots(2,1)
    
        # Get spatial and temporal extents
        x_min, x_max = self.space.mesh.vertices.min(), self.space.mesh.vertices.max()
        t_min, t_max = tt.min(), tt.max()
        
        # extract rho and v
        rho = zz[:,:self.space.dim]
        v = zz[:,self.space.dim:]
        
        # imshow displays data with origin at top-left by default
        # extent: [left, right, bottom, top]
        im_rho = ax_rho.imshow(
            rho,
            aspect='auto',
            origin='lower',
            extent=[x_min, x_max, t_min, t_max],
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            interpolation=interpolation
        )
        
        im_v = ax_v.imshow(
            v,
            aspect='auto',
            origin='lower',
            extent=[x_min, x_max, t_min, t_max],
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            interpolation=interpolation
        )
        
        ax_v.set_ylabel('time $t$')
        ax_rho.set_ylabel('time $t$')
        ax_v.set_xlabel('space $x$')
        ax_rho.tick_params(axis='x', labelbottom=False) # hide xtick labels for rho axis
        plt.colorbar(im_rho, ax=ax_rho, label=r'$\rho(t,x)$', ticks=colorbarticks)
        plt.colorbar(im_v, ax=ax_v, label=r'$v(t,x)$', ticks=colorbarticks)
        
        if savepath is not None:
            fig.tight_layout()
            fig.savefig(savepath + '.pgf')  # save as pgf
            fig.savefig(savepath + '.png')  # save as png
            print(f'figure saved under savepath {savepath} (as pgf and png)')
        
        if title is not None:
            fig.suptitle(title)
    
        fig.tight_layout()
        fig.show()
    
        return fig
    
class QuasilinearWaveReducedOrder(QuasilinearWave):
    
    def __init__(
            self,
            reduced_order: int = 10,
            picklepath: str = None,
            **kwargs,
            ):
        
        super().__init__(**kwargs)
        
        # store reduced order dimension
        self.reduced_order = reduced_order
        
        # simulate the system to obtain snapshot matrix
        T = 0.1
        nt = 501 # fine discretization
        tt = jnp.linspace(0, T, nt)
        degree = 2
        num_quad_nodes = degree
        num_proj_nodes = 2*degree
        picklename = f'{picklepath}_n{degree}_qn{num_quad_nodes}_pn{num_proj_nodes}_M{nt}' # needs to be updated

        try: # try to skip also the evaluation
            with open(f'{picklename}.pickle','rb') as f:
                proj_solution = pickle.load(f)['proj_solution']
            print(f'\tFOM result was loaded')

        except FileNotFoundError: # evaluation was not done before
            s = timer()
            proj_solution = projection_method(
                ebm=QuasilinearWave(**kwargs),
                tt=tt,
                z0=self.initial_condition,
                control=self.default_control,
                degree=degree,
                num_quad_nodes=num_quad_nodes,
                num_proj_nodes=num_proj_nodes,
                debug=False,
                )
            e = timer()
            print(f'[quasilinear wave rom] full order simulation took {e-s:.2f} seconds')

            # save file
            if picklepath is not None: # save at valid path
                with open(f'{picklename}.pickle','wb') as f:
                    pickle.dump({'proj_solution': proj_solution},f)
                print(f'\tFOM result was written')
    
        _, zz, dt_zz = proj_solution['boundaries']
        rho = zz[:,:self.space.dim]
        v = zz[:,self.space.dim:]
        Q_rho = rho.T # snapshot matrix for rho states - time index in second position
        Q_v = rho.T # snapshot matrix for v states - time index in second position
        U_rho, _, _ = jnp.linalg.svd(Q_rho, full_matrices=False)
        U_v, _, _ = jnp.linalg.svd(Q_v, full_matrices=False)
        self.V2 = jnp.vstack((U_rho[:,:self.reduced_order],U_v[:,:self.reduced_order]))
        
        # build reduced order matrix
        self.V1 = self.V3 = jnp.eye(0) # self.dims[0] == self.dims[2] == 0
        self.V = jax.scipy.linalg.block_diag(self.V1, self.V2, self.V3)
        
        # update state dimension
        self.dims = (self.V1.shape[1], self.V2.shape[1], self.V3.shape[1])
        
        # update initial condition
        self.initial_condition = self.V.T @ self.initial_condition
        
        # set rom flag
        self.is_rom = True
    
    def visualize_solution(
            self,
            zz: jnp.ndarray,
            **kwargs,
            ):
        
        Vzz = jnp.einsum('nr,tr->tn', self.V, zz)
        
        return super().visualize_solution(
            zz=Vzz,
            **kwargs,
            )
        
    def hamiltonian(self, z1, z2):
        return super().hamiltonian(self.V1 @ z1, self.V2 @ z2)
    
    def J(self, dt_z1, h2, z3):
        return self.V.T @ super().J(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
    
    def R(self, dt_z1, h2, z3):
        return self.V.T @ super().R(self.V1 @ dt_z1, self.V2 @ h2, self.V3 @ z3)
        
    def B(self, u):
        return self.V.T @ super().B(u)



if __name__ == "__main__":
    
    pass