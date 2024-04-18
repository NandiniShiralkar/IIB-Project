from ._imports import *
from .utils import Euler2fixedpt

class _SSN_Base(nn.Module):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None, device='cpu', dtype=torch.float64):
        super(_SSN_Base, self).__init__()
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni
        
        # Use a boolean tensor to represent 'E' (True) and 'I' (False) neuron types
        self.EI = torch.zeros(self.N, dtype=torch.bool, device=device)
        self.EI[:Ne] = True  # Set the first Ne elements to True representing 'E' neurons
        
        # Convert numpy arrays to PyTorch tensors and move them to the specified device
        self.tau_vec = tau_vec.to(device) if tau_vec is not None else None
        self.W = W.to(device) if W is not None else None

        self.device = device
        self.dtype = dtype

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    def powlaw(self, u):
        return self.k * F.relu(u).pow(self.n)

    def drdt(self, r, inp_vec):
        
        return (-r + self.powlaw(self.W @ r + inp_vec)) / self.tau_vec

    def drdt_multi(self, r, inp_vec):
        """
        Compared to self.drdt allows for inp_vec and r to be
        matrices with arbitrary shape[1]
        """
        # Add an extra dimension to r for broadcasting
        r_expanded = r.unsqueeze(-1).double()
    
        # Compute the matrix multiplication along the last two dimensions
        Wr = torch.matmul(self.W, r_expanded).squeeze(-1)
    
        # Add the input vector to the result
        Wr_plus_inp = Wr + inp_vec.double()
    
        # Apply the power law function
        pow_result = self.powlaw(Wr_plus_inp)
    
        # Compute the derivative
        drdt = (-r + pow_result) / self.tau_vec.double()
    
        return drdt
        
        #return ((-r + self.powlaw(torch.mm(self.W, r) + inp_vec)).transpose(0,1) / self.tau_vec).transpose(0,1)

    def dxdt(self, x, inp_vec):
        """
        Allowing for descendant SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)

    def gains_from_v(self, v):
        return self.n * self.k * F.relu(v).pow(self.n - 1)

    def gains_from_r(self, r):
        return self.n * self.k**(1/self.n) * r.pow(1 - 1/self.n)

    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around rate vector r
        """
        Phi = self.gains_from_r(r)
        return -torch.eye(self.N, device=self.device, dtype=self.dtype) + Phi[:, None] * self.W

    def jacobian(self, DCjacob=None, r=None):
        """
        Dynamic Jacobian for linearisation around rate vector r
        """
        if DCjacob is None:
            assert r is not None, "Either DCjacob or r must be provided."
            DCjacob = self.DCjacobian(r)
        return DCjacob / self.tau_x_vec[:, None]

    def jacobian_eigvals(self, DCjacob=None, r=None):
        Jacob = self.jacobian(DCjacob=DCjacob, r=r)
        return torch.linalg.eigvals(Jacob)

    def inv_G(self, omega, DCjacob, r=None):
        """
        Inverse Green's function at angular frequency omega,
        for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None, "Either DCjacob or r must be provided."
            DCjacob = self.DCjacobian(r)
        return -1j * omega * torch.diag(self.tau_x_vec) - DCjacob
    
    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False, verbose=True):
        """
        Find the fixed point for rate dynamics, given an input vector.
        
        Args:
        inp_vec (torch.Tensor): Input vector.
        r_init (torch.Tensor, optional): Initial guess for rate vector.
        Tmax (float, optional): Maximum time to run the Euler method.
        dt (float, optional): Time step for Euler method.
        xtol (float, optional): Tolerance for convergence.
        PLOT (bool, optional): If True, plot the convergence.
        verbose (bool, optional): If True, print convergence information.
        device (str or torch.device, optional): Device to run the computation on ('cpu' or 'cuda').
        dtype (torch.dtype, optional): Data type of the tensors.

        Returns:
        r_fp (torch.Tensor): Fixed point rate vector.
        CONVG (bool): True if convergence was achieved, False otherwise.
        """
        inp_vec = inp_vec.to(device=self.device, dtype=self.dtype)
        
        if r_init is None:
            r_init = torch.zeros_like(inp_vec,dtype=self.dtype)
        else:
            r_init = r_init.to(device=self.device, dtype=self.dtype)
        
        drdt = lambda r : self.drdt(r, inp_vec)
        if inp_vec.ndim > 1:
            drdt = lambda r : self.drdt_multi(r, inp_vec)
                    
        r_fp, CONVG = Euler2fixedpt(drdt, r_init, Tmax, dt, xtol=xtol, PLOT=PLOT, verbose=verbose, device=self.device, dtype=self.dtype)
        if not CONVG:
            print('Did not reach fixed point.')
            
        return r_fp, CONVG

    def fixed_point(self, inp_vec, x_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False):
        """
        Find the fixed point for the system dynamics, given an input vector.

        Args are similar to `fixed_point_r`.

        x_fp (torch.Tensor): Fixed point state vector.
        CONVG (bool): True if convergence was achieved, False otherwise.
        """
        inp_vec = inp_vec.to(device=self.device, dtype=self.dtype)
        
        if x_init is None:
            x_init = torch.zeros((self.dim,), device=self.device, dtype=self.dtype)
        else:
            x_init = x_init.to(device=self.device, dtype=self.dtype)
            
        dxdt = lambda x : self.dxdt(x, inp_vec)
        x_fp, CONVG = Euler2fixedpt(dxdt, x_init, Tmax, dt, xtol=xtol, PLOT=PLOT, device=self.device, dtype=self.dtype)
        if not CONVG:
            print('Did not reach fixed point.')
            
        return x_fp, CONVG

    def make_noise_cov(self, noise_pars):
        """
        Create the noise covariance matrix based on noise parameters.

        Args:
        noise_pars (object): An object that has attributes 'stdevE' and 'stdevI', representing standard deviations.
        device (str or torch.device, optional): Device to run the computation on ('cpu' or 'cuda').
        dtype (torch.dtype, optional): Data type of the tensors.

        Returns:
        noise_sigsq (torch.Tensor): Noise variances.
        spatl_filt (torch.Tensor): Spatial filter.
        """
        noise_sigsq = torch.cat([
            (noise_pars.stdevE ** 2) * torch.ones(self.Ne, device=self.device, dtype=self.dtype),
            (noise_pars.stdevI ** 2) * torch.ones(self.Ni, device=self.device, dtype=self.dtype)
        ])
        spatl_filt = torch.tensor(1, device=self.device, dtype=self.dtype)

        return noise_sigsq, spatl_filt