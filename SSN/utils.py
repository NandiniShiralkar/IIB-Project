from _imports import *

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, inds=None, verbose=True, device='cpu', dtype=torch.float32):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    Args:
    dxdt (function): A function handle giving the right-hand side function of the dynamical system.
    x_initial (torch.Tensor): Initial condition for state variables (a column vector).
    Tmax (float): Maximum time to which it would run the Euler (same units as dt, e.g., ms).
    dt (float): Time step of Euler.
    xtol (float, optional): Tolerance in relative change in x for determining convergence.
    xmin (float, optional): For x(i)<xmin, it checks convergence based on absolute change, which must be smaller than xtol*xmin.
    Tmin (float, optional): Minimum time for convergence check.
    PLOT (bool, optional): If True, plot the convergence of some component.
    inds (list or torch.Tensor, optional): Indices of x (state-vector) to plot.
    verbose (bool, optional): If True, print convergence information.
    device (str or torch.device, optional): Device to run the computation on ('cpu' or 'cuda').
    dtype (torch.dtype, optional): Data type of the tensors.

    Returns:
    xvec (torch.Tensor): Found fixed point solution.
    CONVG (bool): True if determined converged, False if not.
    """

    x_initial = x_initial.to(device=device, dtype=dtype)

    if PLOT:
        if inds is None:
            N = x_initial.numel()
            inds = torch.tensor([int(N/4), int(3*N/4)], device=device, dtype=torch.long)
        xplot = x_initial[inds].view(-1, 1)

    Nmax = int(round(Tmax / dt))
    Nmin = int(round(Tmin / dt)) if Tmax > Tmin else int(Nmax / 2)
    xvec = x_initial.clone()
    CONVG = False
    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx
        if PLOT:
            xplot = torch.cat((xplot, xvec[inds].view(-1, 1)), dim=1)

        if n > Nmin:
            rel_change = torch.abs(dx / torch.maximum(torch.tensor(xmin, device=xvec.device, dtype=xvec.dtype), torch.abs(xvec)))
            if rel_change.max() < xtol:
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG:  # n == Nmax
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, rel_change.max().item(), xtol))

    if PLOT:
        plt.figure(244459)
        plt.plot(torch.arange(n+2).to(dtype=dtype, device=device) * dt, xplot.cpu().numpy(), 'o-')

    return xvec, CONVG