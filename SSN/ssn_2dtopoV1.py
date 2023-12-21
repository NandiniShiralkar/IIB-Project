from _imports import *
from ssn_base import _SSN_Base

class SSN2DTopoV1(_SSN_Base,nn.Module):
    _Lring = 180

    def __init__(self, n, k, tauE, tauI, grid_pars, conn_pars, **kwargs):
        Ne = Ni = grid_pars.gridsize_Nx ** 2
        tau_vec = torch.cat([tauE * torch.ones(Ne), tauI * torch.ones(Ni)])
        super(SSN2DTopoV1, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs)
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self._make_maps(grid_pars)
        if conn_pars is not None:
            self.make_W(**conn_pars)

    @property
    def neuron_params(self):
        # Return a dictionary of neuron parameters
        return dict(n=self.n, k=self.k, tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])

    @property
    def maps_vec(self):
        # Combine x, y, and orientation vectors and return them as a matrix
        return torch.stack([self.x_vec, self.y_vec, self.ori_vec]).T

    @property
    def center_inds(self):
        """ Indices of center-E and center-I neurons """
        # Find the indices where x and y coordinates are both zero
        return torch.where((self.x_vec == 0) & (self.y_vec == 0))[0]

    @property
    def x_vec_degs(self):
        # Convert x coordinates from mm to degrees
        return self.x_vec / self.grid_pars.magnif_factor

    @property
    def y_vec_degs(self):
        # Convert y coordinates from mm to degrees
        return self.y_vec / self.grid_pars.magnif_factor

    def xys2inds(self, xys=[[0, 0]], units="degree"):
        """
        Indices of E and I neurons at location (x,y) (by default in degrees).
        Args:
            xys: List or array-like object containing xy coordinates.
            units: Specifies unit for xys. By default, "degree" of visual angle.
        Returns:
            inds: Tensor of shape (2, len(xys)), where inds[0] contains indices of E neurons
                  and inds[1] contains indices of I neurons.
        """
        inds = []
        for xy in xys:
            if units == "degree":  # Convert to mm
                xy = self.grid_pars.magnif_factor * torch.tensor(xy)
            distsq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            inds.append([torch.argmin(distsq[:self.Ne]), self.Ne + torch.argmin(distsq[self.Ne:])])
        return torch.tensor(inds).T

    def xys2Emapinds(self, xys=[[0, 0]], units="degree"):
        """
        (i,j) of E neurons at location (x,y) (by default in degrees).
        Args:
            xys: List or array-like object containing xy coordinates.
            units: Specifies unit for xys. By default, "degree" of visual angle.
        Returns:
            map_inds: Tensor of shape (2, len(xys)), where inds[0] contains row indices of E neurons in map
                      and inds[1] contains column indices of E neurons in map.
        """
        def vecind2mapind(i):
            return torch.tensor([i % self.grid_pars.gridsize_Nx, i // self.grid_pars.gridsize_Nx])
        
        return vecind2mapind(self.xys2inds(xys)[0])

    def vec2map(self, vec):
        """
        Reshape a 1-dimensional tensor to a 2-dimensional map or a pair of maps.
        Args:
            vec: 1-dimensional tensor.
        Returns:
            map: If vec corresponds to E neurons, a single 2D map is returned.
                 If vec corresponds to all neurons, a tuple of E and I maps is returned.
        """
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Ne:
            map = vec.view(Nx, Nx)
        elif len(vec) == self.N:
            map = (vec[:self.Ne].reshape((Nx, Nx)), vec[self.Ne:].reshape((Nx, Nx)))
        return map

    def _make_maps(self, grid_pars=None):
        """
        Create retinotopic and orientation maps.
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        self._make_retinmap()
        self._make_orimap()

        return self.x_map, self.y_map, self.ori_map

    def _make_retinmap(self, grid_pars=None):
        """
        Create square grid of locations with X and Y retinotopic maps.
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        if not hasattr(grid_pars, "gridsize_mm"):
            self.grid_pars.gridsize_mm = grid_pars.gridsize_deg * grid_pars.magnif_factor

        Lx = Ly = self.grid_pars.gridsize_mm
        Nx = Ny = grid_pars.gridsize_Nx
        dx = dy = Lx / (Nx - 1)

        self.grid_pars.dx = dx  # in mm
        self.grid_pars.dy = dy  # in mm

        xs = torch.linspace(0, Lx, Nx)
        ys = torch.linspace(0, Ly, Ny)
        X, Y = torch.meshgrid(xs - xs[len(xs) // 2], ys - ys[len(ys) // 2])
        Y = -Y  # without this Y decreases going upwards

        self.x_map = X
        self.y_map = Y
        self.x_vec = X.ravel().repeat(2)
        self.y_vec = Y.ravel().repeat(2)
        return self.x_map, self.y_map

    def _make_orimap(self, hyper_col=None, nn=30, X=None, Y=None):
        """
        Makes the orientation map for the grid, by superposition of plane-waves.
        """
        if hyper_col is None:
            hyper_col = self.grid_pars.hyper_col
        else:
            self.grid_pars.hyper_col = hyper_col

        X = self.x_map if X is None else X
        Y = self.y_map if Y is None else Y

        z = torch.zeros_like(X, dtype=torch.complex128)
        for j in range(nn):
            kj = torch.tensor([np.cos(j * np.pi / nn), np.sin(j * np.pi / nn)]) * 2 * np.pi / (hyper_col)
            sj = 2 * np.random.choice([-1, 1])  # random number that's either + or -1
            phij = np.random.rand() * 2 * np.pi

            tmp = (X * kj[0] + Y * kj[1]) * sj + phij
            z = z + torch.exp(1j * tmp)

        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (torch.angle(z) + np.pi) * SSN2DTopoV1._Lring / (2 * np.pi)
        self.ori_vec = self.ori_map.ravel().repeat(2)
        return self.ori_map

    def _make_distances(self, PERIODIC):
        """
        Compute distances between neurons in both space and orientation.
        """
        Lx = Ly = self.grid_pars.gridsize_mm

        def absdiff_ring(d_x, L):
            return torch.minimum(torch.abs(d_x), L - torch.abs(d_x))

        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: torch.abs(d_x)

        xs = self.x_vec.reshape(2, self.Ne, 1)  # (cell-type, grid-location, None)
        ys = self.y_vec.reshape(2, self.Ne, 1)  # (cell-type, grid-location, None)
        oris = self.ori_vec.reshape(2, self.Ne, 1)  # (cell-type, grid-location, None)

        xy_dist = torch.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = absdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1._Lring)

        self.xy_dist = xy_dist
        self.ori_dist = ori_dist

        return xy_dist, ori_dist

    def make_W(self, J_2x2, s_2x2, p_local, sigma_oris=45, Jnoise=0,
               Jnoise_GAUSSIAN=False, MinSyn=1e-4, CellWiseNormalized=True,
               PERIODIC=True):  # , prngKey=0):
        """
        make the full recurrent connectivity matrix W
        :param J_2x2: total strength of weights of different pre/post cell-type
        :param s_2x2: ranges of weights between different pre/post cell-type
        :param p_local: relative strength of local parts of E projections
        :param sigma_oris: range of weights in terms of preferred orientation difference
        :param Jnoise: amount of noise to add
        :param Jnoise_GAUSSIAN: if True, noise is Gaussian, otherwise it's uniform
        :param MinSyn: minimum synaptic weight
        :param CellWiseNormalized: if True, normalize weights cell-wise
        :param PERIODIC: if True, use periodic boundary conditions
        :return: connectivity matrix W
        """
        conn_pars = locals()
        conn_pars.pop("self")
        self.conn_pars = conn_pars


        if hasattr(self, "xy_dist") and hasattr(self, "ori_dist"):
            xy_dist = self.xy_dist
            ori_dist = self.ori_dist
        else:
            xy_dist, ori_dist = self._make_distances(PERIODIC)

        # Check if sigma_oris is a scalar
        sigma_oris = torch.tensor(sigma_oris)
        if sigma_oris.numel() == 1:
            sigma_oris = sigma_oris * torch.ones((2, 2))

        #if not torch.is_tensor(sigma_oris):
        # If it's not a tensor, we make it into a 2x2 tensor
            #sigma_oris = torch.full((2, 2), fill_value=sigma_oris, device = self.device)

        p_local = torch.tensor(p_local)
        if p_local.numel() == 1:
            p_local = p_local * torch.ones(2)

        #if not torch.is_tensor(p_local) or p_local.numel() == 1:
           # p_local = torch.full((2,), fill_value=p_local, device=self.device)

        shape = self.xy_dist.shape
        Wblks = [[torch.zeros(shape, device=self.device), torch.zeros(shape, device=self.device)],
                 [torch.zeros(shape, device=self.device), torch.zeros(shape, device=self.device)]]

        # loop over post- (a) and pre-synaptic (b) cell-types
        for a in range(2):
            for b in range(2):
                if b == 0:  # E projections
                    W = torch.exp(-xy_dist / s_2x2[a, b] - ori_dist ** 2 / (2 * sigma_oris[a, b] ** 2))
                elif b == 1:  # I projections
                    W = torch.exp(-xy_dist ** 2 / (2 * s_2x2[a, b] ** 2) - ori_dist ** 2 / (2 * sigma_oris[a, b] ** 2))

                if Jnoise > 0:  # add some noise
                    if Jnoise_GAUSSIAN:
                        jitter = torch.randn_like(W)
                    else:
                        jitter = 2 * torch.rand_like(W) - 1
                    W = (1 + Jnoise * jitter) * W

                # sparsify (set small weights to zero)
                W = torch.where(W < MinSyn, torch.zeros_like(W), W)

                # row-wise normalize
                tW = torch.sum(W, dim=1, keepdim=True)
                if not CellWiseNormalized:
                    tW = tW.mean()
                W = W / tW

                #tW = torch.sum(W, dim=1)
                #if not CellWiseNormalized:
                   # tW = torch.mean(tW)
                #W = W / tW.unsqueeze(1)

                # for E projections, add the local part
                if b == 0:
                    W = p_local[a] * torch.eye(*W.shape, device=self.device) + (1 - p_local[a]) * W

                Wblks[a][b] = J_2x2[a, b] * W

        self.W = torch.cat([
                torch.cat([Wblks[0][0], Wblks[0][1]], dim=1), 
                torch.cat([Wblks[1][0], Wblks[1][1]], dim=1)
            ], dim=0)
        self.W = self.W.float()  # Convert W to float32

        return self.W

    def _make_inp_ori_dep(self, ONLY_E=False, ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1):
        """
        Makes the orientation dependence factor for grating or Gabor stimuli (a la Ray & Maunsell 2010)
        :param ONLY_E: if True, only make the E-part of the input vector
        :param ori_s: stimulus orientation
        :param sig_ori_EF: orientation tuning-width of E cells
        :param sig_ori_IF: orientation tuning-width of I cells
        :param gE: amplitude of E part
        :param gI: amplitude of I part
        :return: orientation factor
        """
        if ori_s is None:
            # set stim ori to pref ori of grid center E cell (same as I cell)
            center_E_indices = (self.x_vec == 0) & (self.y_vec == 0) & self.EI
            ori_s = self.ori_vec[center_E_indices]
        if sig_ori_IF is None:
            sig_ori_IF = sig_ori_EF

        distsq = lambda x: torch.min(torch.abs(x), self._Lring - torch.abs(x)) ** 2
        dori = self.ori_vec - ori_s
        if not ONLY_E:
            ori_fac = torch.cat((
                gE * torch.exp(-distsq(dori[:self.Ne]) / (2 * sig_ori_EF ** 2)),
                gI * torch.exp(-distsq(dori[self.Ne:]) / (2 * sig_ori_IF ** 2))
            ))
        else:
            ori_fac = gE * torch.exp(-distsq(dori[:self.Ne]) / (2 * sig_ori_EF ** 2))

        return ori_fac

    def make_grating_input(self, radius_s, sigma_RF=0.4, ONLY_E=False,
                           ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        Make grating external input
        :param radius_s: radius of the stimulus
        :param sigma_RF: edge-fall-off scale
        :param ONLY_E: if True, only make the E-part of the input vector
        :param ori_s: stimulus orientation
        :param sig_ori_EF: orientation tuning-width of E cells
        :param sig_ori_IF: orientation tuning-width of I cells
        :param gE: amplitude of E part
        :param gI: amplitude of I part
        :param contrast: contrast of the stimulus
        :return: input vector
        """
        # Make the orientation dependence factor
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # Make the spatial envelope
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        M = self.Ne if ONLY_E else self.N
        r_vec = torch.sqrt(self.x_vec_degs[:M] ** 2 + self.y_vec_degs[:M] ** 2)
        spat_fac = sigmoid((radius_s - r_vec) / sigma_RF)

        return contrast * ori_fac * spat_fac

    def make_gabor_input(self, sigma_Gabor=0.5, ONLY_E=False,
                         ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        Make the Gabor stimulus (a la Ray & Maunsell 2010) centered on the
        grid-center, with sigma "sigma_Gabor",
        with orientation "ori_s",
        with the orientation tuning-width of E and I parts given by "sig_ori_EF"
        and "sig_ori_IF", respectively, and with amplitude (maximum) of the E and I parts,
        given by "contrast * gE" and "contrast * gI", respectively.
        """
        # Make the orientation dependence factor
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # Make the spatial envelope
        gaussian = lambda x: torch.exp(- x**2 / 2)
        M = self.Ne if ONLY_E else self.N
        r_vec = torch.sqrt(self.x_vec_degs[:M]**2 + self.y_vec_degs[:M]**2)
        spat_fac = gaussian(r_vec / sigma_Gabor)

        return contrast * ori_fac * spat_fac

    def make_eLFP_from_inds(self, LFPinds):
        """
        Makes a single LFP electrode signature (normalized spatial weight
        profile), given the (vectorized) indices of recorded neurons (LFPinds).

        OUT: e_LFP with shape (self.N,)
        """
        if LFPinds is None:
            LFPinds = [0]
        e_LFP = 1 / len(LFPinds) * torch.isin(torch.arange(self.N, device=self.device), torch.tensor(LFPinds, device=self.device)).float()
        return e_LFP.float()

    def make_eLFP_from_xy(self, probe_xys, LFPradius=0.2, unit_xys="degree", unit_rad="mm"):
        """
        Makes 1 or multiple LFP electrodes signatures (normalized spatial weight
        profile over E cells), given the (x,y) retinotopic coordinates of LFP probes.

        IN: probe_xys: shape (#probes, 2). Each row is the (x,y) coordinates of
            a probe/electrode (by default given in degrees of visual angle)
            LFPradius: positive scalar. radius/range of LFP (by default given in mm)
            unit_xys: either "degree" or "mm", unit of LFP_xys
            unit_rad: either "degree" or "mm", unit of LFPradius
        OUT: e_LFP: shape (self.N, #probes) = (self.N, LFP.xys.shape[0])
            Each column is the normalized spatial profile of one probe.
        """
        if unit_rad == "degree":
            LFPradius = self.grid_pars.magnif_factor * LFPradius

        e_LFP = []
        for xy in probe_xys:
            if unit_xys == "degree":  # Convert to mm
                xy = self.grid_pars.magnif_factor * torch.tensor(xy, device=self.device)

            dist_sq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            e_LFP.append(1.0 * (self.EI & (dist_sq < LFPradius**2)))

        return torch.stack(e_LFP).T

    def run_and_visualise_dynamics(self, inp_vec, total_time, dt):
        
        """
        Run the dynamics from a fixed point and visualise the neuron states over time.

        Args:
        inp_vec (torch.Tensor): Input vector to find the fixed point.
        total_time (float): Total time to run the simulation.
        dt (float): Time step for the simulation.
        """
        # Find the fixed point for the given input
        #r_fixed, _ = self.fixed_point(inp_vec)

        # Run the dynamics starting from the fixed point
        num_steps = int(total_time / dt)
        neuron_states = torch.zeros((num_steps, self.N), device=self.device)
        #r = r_fixed
        r = torch.zeros((self.N*2, 1), device=self.device)

        for step in range(num_steps):
            # Calculate rate of change of neuron states
            drdt = self.drdt(r, inp_vec)

            # Euler integration
            r += dt * drdt

            neuron_states[step, :] = r

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(neuron_states.cpu().numpy().T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Neuron State')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')
        plt.title('Neuron States Over Time')
        plt.show()