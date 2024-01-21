import numpy as np

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=False, silent=False, Tfrac_CV=0):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot
    verbose: if True print convergence criteria even if passed (function always prints out a warning if it doesn't converge).
    Tfrac_var: if not zero, maximal temporal CV (coeff. of variation) of state vector components, over the final
               Tfrac_CV fraction of Euler timesteps, is calculated and printed out.
               
    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    if PLOT:
        if inds is None:
            x_dim = x_initial.size
            inds = [int(x_dim/4), int(3*x_dim/4)]
        xplot = x_initial.flatten()[inds][:,None]

    Nmax = int(np.round(Tmax/dt))
    Nmin = int(np.round(Tmin/dt)) if Tmax > Tmin else int(Nmax/2)
    xvec = x_initial
    CONVG = False

    if Tfrac_CV > 0:
        xmean = np.zeros_like(xvec)
        xsqmean = np.zeros_like(xvec)
        Nsamp = 0

    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            xplot = np.hstack((xplot, xvec.flatten()[inds][:,None]))
        
        if Tfrac_CV > 0 and n >= (1-Tfrac_CV) * Nmax:
            xmean = xmean + xvec
            xsqmean = xsqmean + xvec**2
            Nsamp = Nsamp + 1

        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))

        if Tfrac_CV > 0:
            xmean = xmean/Nsamp
            xvec_SD = np.sqrt(xsqmean/Nsamp - xmean**2)
            # CV = xvec_SD / xmean
            # CVmax = CV.max()
            CVmax = xvec_SD.max() / xmean.max()
            print(f"max(SD)/max(mean) of state vector in the final {Tfrac_CV:.2} fraction of Euler steps was {CVmax:.5}")

        #mybeep(.2,350)
        #beep

    if PLOT:
        import matplotlib.pyplot as plt
        plt.figure(244459)
        plt.plot(np.arange(n+2)*dt, xplot.T, 'o-')

    return xvec, CONVG


class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni
        self.EI = np.chararray((self.N,), itemsize=1)
        self.EI[:Ne] = b"E"
        self.EI[Ne:] = b"I"
        if tau_vec is not None:
            self.tau_vec = tau_vec # rate time-consants of neurons. shape: (N,)
        # elif  not hasattr(self, "tau_vec"):
        #     self.tau_vec = np.random.rand(N) * 20 # in ms
        if W is not None:
            self.W = W # connectivity matrix. shape: (N, N)
        # elif  not hasattr(self, "W"):
        #     W = np.random.rand(N,N) / np.sqrt(self.N)
        #     sign_vec = np.hstack(np.ones(self.Ne), -np.ones(self.Ni))
        #     self.W = W * sign_vec[None, :] # to respect Dale


    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    @property
    def dim(self):
        return self.N

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_vec


    def powlaw(self, u):
        return  self.k * np.maximum(0,u)**self.n


    def drdt(self, r, inp_vec):
        return ( -r + self.powlaw(self.W @ r + inp_vec) ) / self.tau_vec


    def drdt_multi(self, r, inp_vec):
        """
        Compared to self.drdt allows for inp_vec and r to be
        matrices with arbitrary shape[1]
        """
        return (( -r + self.powlaw(self.W @ r + inp_vec) ).T / self.tau_vec ).T


    def dxdt(self, x, inp_vec):
        """
        allowing for descendent SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)


    def gains_from_v(self, v):
        return self.n * self.k * np.maximum(0,v)**(self.n-1)


    def gains_from_r(self, r):
        return self.n * self.k**(1/self.n) * r**(1-1/self.n)


    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around rate vector r
        """
        Phi = self.gains_from_r(r)
        return -np.eye(self.N) + Phi[:, None] * self.W


    def jacobian(self, DCjacob=None, r=None):
        """
        dynamic Jacobian for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return DCjacob / self.tau_x_vec[:, None] # equivalent to np.diag(tau_x_vec) * DCjacob


    def jacobian_eigvals(self, DCjacob=None, r=None):
        Jacob = self.jacobian(DCjacob=DCjacob, r=r)
        return np.linalg.eigvals(Jacob)


    def inv_G(self, omega, DCjacob, r=None):
        """
        inverse Green's function at angular frequency omega,
        for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return -1j*omega * np.diag(self.tau_x_vec) - DCjacob


    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False, verbose=False, silent=False):
        if r_init is None:
            r_init = np.zeros(inp_vec.shape) # np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)
        if inp_vec.ndim > 1:
            drdt = lambda r : self.drdt_multi(r, inp_vec)
        r_fp, CONVG = Euler2fixedpt(drdt, r_init, Tmax, dt, xtol=xtol, PLOT=PLOT, verbose=verbose, silent=silent)
        if not CONVG and not silent:
            print('Did not reach fixed point.')
        #else:
        #    return r_fp
        return r_fp, CONVG


    def fixed_point(self, inp_vec, x_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False, verbose=False, silent=False):
        if x_init is None:
            x_init = np.zeros((self.dim,))
        dxdt = lambda x : self.dxdt(x, inp_vec)
        x_fp, CONVG = Euler2fixedpt(dxdt, x_init, Tmax, dt, xtol=xtol, PLOT=PLOT, verbose=verbose, silent=silent)
        if not CONVG and not silent:
            print('Did not reach fixed point.')
        #else:
        #    return x_fp
        return x_fp, CONVG


    def make_noise_cov(self, noise_pars):
        # the script assumes independent noise to E and I, and spatially uniform magnitude of noise
        noise_sigsq = np.hstack( (noise_pars.stdevE**2 * np.ones(self.Ne),
                                  noise_pars.stdevI**2 * np.ones(self.Ni)) )
        spatl_filt = np.array(1)

        return noise_sigsq, spatl_filt

class SSN2DTopoV1(_SSN_Base):
    _Lring = 180

    def __init__(self, n, k, tauE, tauI, grid_pars, conn_pars, ori_map=None, **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])

        super(SSN2DTopoV1, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self._make_maps(grid_pars, ori_map)
        if conn_pars is not None: # conn_pars = None allows for ssn-object initialization without a W
            self.make_W(**conn_pars)


    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    @property
    def maps_vec(self):
        return np.vstack([self.x_vec, self.y_vec, self.ori_vec]).T

    @property
    def x_vec_degs(self):
        return self.x_vec / self.grid_pars.magnif_factor

    @property
    def y_vec_degs(self):
        return self.y_vec / self.grid_pars.magnif_factor

    @property
    def center_inds(self):
        """ indices of center-E and center-I neurons """
        return np.where((self.x_vec==0) & (self.y_vec==0))[0]


    def xys2inds(self, xys=[[0,0]], units="degree"):
        """
        indices of E and I neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            inds: shape = (2, len(xys)), inds[0] = vector-indices of E neurons
                                         inds[1] = vector-indices of I neurons
        """
        inds = []
        for xy in xys:
            if units == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            distsq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            inds.append([np.argmin(distsq[:self.Ne]), self.Ne + np.argmin(distsq[self.Ne:])])
        return np.asarray(inds).T


    def xys2Emapinds(self, xys=[[0,0]], units="degree"):
        """
        (i,j) of E neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            map_inds: shape = (2, len(xys)), inds[0] = row_indices of E neurons in map
                                         inds[1] = column-indices of E neurons in map
        """
        vecind2mapind = lambda i: np.array([i % self.grid_pars.gridsize_Nx,
                                            i // self.grid_pars.gridsize_Nx])
        return vecind2mapind(self.xys2inds(xys)[0])


    def vec2map(self, vec):
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Ne:
            map = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.N:
            map = (np.reshape(vec[:self.Ne], (Nx, Nx)),
                   np.reshape(vec[self.Ne:], (Nx, Nx)))
        return map


    def _make_maps(self, grid_pars=None, ori_map=None):
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        self._make_retinmap()
        self.ori_map = self._make_orimap() if ori_map is None else ori_map

        return self.x_map, self.y_map, self.ori_map


    def _make_retinmap(self, grid_pars=None):
        """
        make square grid of locations with X and Y retinotopic maps
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars
        if not hasattr(grid_pars, "gridsize_mm"):
            self.grid_pars.gridsize_mm = grid_pars.gridsize_deg * grid_pars.magnif_factor
        Lx = Ly = self.grid_pars.gridsize_mm
        Nx = Ny = grid_pars.gridsize_Nx
        dx = dy = Lx/(Nx - 1)
        self.grid_pars.dx = dx # in mm
        self.grid_pars.dy = dy # in mm

        xs = np.linspace(0, Lx, Nx)
        ys = np.linspace(0, Ly, Ny)
        [X, Y] = np.meshgrid(xs - xs[len(xs)//2], ys - ys[len(ys)//2]) # doing it this way, as opposed to using np.linspace(-Lx/2, Lx/2, Nx) (for which this fails for even Nx), guarantees that there is always a pixel with x or y == 0
        Y = -Y # without this Y decreases going upwards

        self.x_map = X
        self.y_map = Y
        self.x_vec = np.tile(X.ravel(), (2,))
        self.y_vec = np.tile(Y.ravel(), (2,))
        return self.x_map, self.y_map


    def _make_orimap(self, hyper_col=None, nn=30, X=None, Y=None):
        '''
        Makes the orientation map for the grid, by superposition of plane-waves.
        hyper_col = hyper column length for the network in retinotopic degrees
        nn = (30 by default) # of planewaves used to construct the map

        Outputs/side-effects:
        OMap = self.ori_map = orientation preference for each cell in the network
        self.ori_vec = vectorized OMap
        '''
        if hyper_col is None:
             hyper_col = self.grid_pars.hyper_col
        else:
             self.grid_pars.hyper_col = hyper_col
        X = self.x_map if X is None else X
        Y = self.y_map if Y is None else Y

        z = np.zeros_like(X)
        for j in range(nn):
            kj = np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)]) * 2*np.pi/(hyper_col)
            sj = 2 * np.random.randint(0, 2)-1 #random number that's either + or -1.
            phij = np.random.rand()*2*np.pi

            tmp = (X*kj[0] + Y*kj[1]) * sj + phij
            z = z + np.exp(1j * tmp)

        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (np.angle(z) + np.pi) * SSN2DTopoV1._Lring/(2*np.pi)
        # #for debugging/testing:
        # self.ori_map = 180 * (self.y_map - self.y_map.min())/(self.y_map.max() - self.y_map.min())
        # self.ori_map[self.ori_map.shape[0]//2+1:,:] = 180

        self.ori_vec = np.tile(self.ori_map.ravel(), (2,))

        return self.ori_map


    def _make_distances(self, PERIODIC):
        Lx = Ly = self.grid_pars.gridsize_mm
        absdiff_ring = lambda d_x, L: np.minimum(np.abs(d_x), L - np.abs(d_x))
        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: np.abs(d_x)
        xs = np.reshape(self.x_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        ys = np.reshape(self.y_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        oris = np.reshape(self.ori_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        # to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
        xy_dist = np.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = absdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1._Lring)
        self.xy_dist = xy_dist
        self.ori_dist = ori_dist

        return xy_dist, ori_dist


    def make_W(self, J_2x2, s_2x2, p_local, sigma_oris=45, PERIODIC=True, Jnoise=0,
                Jnoise_GAUSSIAN=False, MinSyn=1e-4, CellWiseNormalized=True): #, prngKey=0):
        """
        make the full recurrent connectivity matrix W
        In:
         J_2x2 = total strength of weights of different pre/post cell-type
         s_2x2 = ranges of weights between different pre/post cell-type
         p_local = relative strength of local parts of E projections
         sigma_oris = range of wights in terms of preferred orientation difference

        Output/side-effects:
        self.W
        """
        # set self.conn_pars to the dictionary of inputs to make_W
        conn_pars = locals()
        conn_pars.pop("self")
        self.conn_pars = conn_pars

        if hasattr(self, "xy_dist") and hasattr(self, "ori_dist"):
            xy_dist = self.xy_dist
            ori_dist = self.ori_dist
        else:
            xy_dist, ori_dist = self._make_distances(PERIODIC)

        if np.isscalar(s_2x2): s_2x2 = s_2x2 * np.ones((2,2))

        if np.isscalar(sigma_oris): sigma_oris = sigma_oris * np.ones((2,2))

        if np.isscalar(p_local) or len(p_local) == 1:
            p_local = np.asarray(p_local) * np.ones(2)

        Wblks = [[1,1],[1,1]]
        # loop over post- (a) and pre-synaptic (b) cell-types
        for a in range(2):
            for b in range(2):
                if b == 0: # E projections
                    W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2))
                elif b == 1: # I projections
                    W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2))

                if Jnoise > 0: # add some noise
                    if Jnoise_GAUSSIAN:
                        jitter = np.random.standard_normal(W.shape)
                    else:
                        jitter = 2* np.random.random(W.shape) - 1
                    W = (1 + Jnoise * jitter) * W

                # sparsify (set small weights to zero)
                W = np.where(W < MinSyn, 0, W) # what's the point of this if not using sparse matrices

                # normalize (do it row-by-row if CellWiseNormalized, such that all row-sums are 1
                #            -- other wise only the average row-sum is 1)
                sW = np.sum(W, axis=1)
                if CellWiseNormalized:
                    W = W / sW[:, None]
                else:
                    W = W / sW.mean()

                # for E projections, add the local part
                # NOTE: this doesn't perturb the above normalization: convex combination of two "probability" vecs
                if b == 0:
                    W = p_local[a] * np.eye(*W.shape) + (1-p_local[a]) * W

                Wblks[a][b] = J_2x2[a, b] * W

        self.W = np.block(Wblks)
        return self.W


    def _make_inp_ori_dep(self, ONLY_E=False, ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1):
        """
        makes the orintation dependence factor for grating or Gabor stimuli
        (a la Ray & Maunsell 2010)
        """
        if ori_s is None:  # set stim ori to pref ori of grid center E cell (same as I cell)
            ori_s = self.ori_vec[(self.x_vec==0) & (self.y_vec==0) & (self.EI==b"E")]
            print(ori_s)
        if sig_ori_IF is None:
            sig_ori_IF = sig_ori_EF

        distsq = lambda x: np.minimum(np.abs(x), SSN2DTopoV1._Lring - np.abs(x))**2
        dori = self.ori_vec - ori_s
        print("ORI_vec: ",self.ori_vec)
        if not ONLY_E:
            ori_fac = np.hstack((gE * np.exp(-distsq(dori[:self.Ne])/(2* sig_ori_EF**2)),
                                 gI * np.exp(-distsq(dori[self.Ne:])/(2* sig_ori_IF**2))))
        else:
            ori_fac = gE * np.exp(-distsq(dori[:self.Ne])/(2* sig_ori_EF**2))

        return ori_fac


    def make_grating_input(self, radius_s, sigma_RF=0.4, ONLY_E=False,
            ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        make grating external input centered on the grid-center, with radius "radius",
        with edge-fall-off scale "sigma_RF", with orientation "ori_s",
        with the orientation tuning-width of E and I parts given by "sig_ori_EF"
        and "sig_ori_IF", respectively, and with amplitue (maximum) of the E and I parts,
        given by "contrast * gE" and "contrast * gI", respectively.
        If ONLY_E=True, it only makes the E-part of the input vector.
        """
        # make the orintation dependence factor:
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # make the spatial envelope:
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        M = self.Ne if ONLY_E else self.N
        r_vec = np.sqrt(self.x_vec_degs[:M]**2 + self.y_vec_degs[:M]**2)
        spat_fac = sigmoid((radius_s - r_vec)/sigma_RF)

        return contrast * ori_fac * spat_fac


    def make_gabor_input(self, sigma_Gabor=0.5, ONLY_E=False,
            ori_s=None, sig_ori_EF=32, sig_ori_IF=None, gE=1, gI=1, contrast=1):
        """
        make the Gabor stimulus (a la Ray & Maunsell 2010) centered on the
        grid-center, with sigma "sigma_Gabor",
        with orientation "ori_s",
        with the orientation tuning-width of E and I parts given by "sig_ori_EF"
        and "sig_ori_IF", respectively, and with amplitue (maximum) of the E and I parts,
        given by "contrast * gE" and "contrast * gI", respectively.
        """
        # make the orintation dependence factor:
        ori_fac = self._make_inp_ori_dep(ONLY_E, ori_s, sig_ori_EF, sig_ori_IF, gE, gI)

        # make the spatial envelope:
        gaussian = lambda x: np.exp(- x**2 / 2)
        M = self.Ne if ONLY_E else self.N
        r_vec = np.sqrt(self.x_vec_degs[:M]**2 + self.y_vec_degs[:M]**2)
        spat_fac = gaussian(r_vec/sigma_Gabor)

        return contrast * ori_fac * spat_fac

    # TODO:
    # def make_noise_cov(self, noise_pars):
    #     # the script assumes independent noise to E and I, and spatially uniform magnitude of noise
    #     noise_sigsq = np.hstack( (noise_pars.stdevE**2 * np.ones(self.Ne),
    #                            noise_pars.stdevI**2 * np.ones(self.Ni)) )
    #
    #     spatl_filt = ...


    def make_eLFP_from_inds(self, LFPinds):
        """
        makes a single LFP electrode signature (normalized spatial weight
        profile), given the (vectorized) indices of recorded neurons (LFPinds).

        OUT: e_LFP with shape (self.N,)
        """
        # LFPinds was called LFPrange in my MATLAB code
        if LFPinds is None:
            LFPinds = [0]
        e_LFP = 1/len(LFPinds) * np.isin(np.arange(self.N), LFPinds) # assuming elements of LFPinds are all smaller than self.Ne, e_LFP will only have 1's on E elements
        # eI = 1/len(LFPinds) * np.isin(np.arange(self.N) - self.Ne, LFPinds) # assuming elements of LFPinds are all smaller than self.Ne, e_LFP will only have 1's on I elements

        return e_LFP


    def make_eLFP_from_xy(self, probe_xys, LFPradius=0.2, unit_xys="degree", unit_rad="mm"):
        """
        makes 1 or multiple LFP electrodes signatures (normalized spatial weight
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
            if unit_xys == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            e_LFP.append(1.0 * ( (self.EI == b"E") &
            (LFPradius**2 > (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2)))

        return np.asarray(e_LFP).T


