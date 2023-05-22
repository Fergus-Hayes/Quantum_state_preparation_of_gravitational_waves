from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
import qiskit_tools as qt
from remez import get_bound_coeffs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, argparse

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(10,10)

def optimize_coeffs_qubits(func, nx, nlab, nintx, ncut0, ncut1, nsig0=4, nsig1=4, norder=1, phase=True):

    def round_sig(xs, sigfig=0):
        if np.array(xs).ndim==0:
            xs = np.array([xs])
        rxs = []
        for x in xs:
            if x!=0.:
                rxs.append(np.round(x, sigfig-int(np.floor(np.log10(np.abs(x))))))
            else:
                rxs.append(0.)
        rxs = np.array(rxs)
        return rxs

    xmax = np.power(2,nintx) - np.power(2,nintx-nx)
    xmin = 0.
    xs = np.linspace(xmin,xmax,2**(nx))

    Nbounds = 2**nlab

    ############ Set piecewise polynomial bounds #################

    bounds_ = np.linspace(xmin, xmax, Nbounds+1)

    bounds__ = []
    for bound in bounds_:
        bounds__.append(qt.bin_to_dec(qt.my_binary_repr(bound, n=nx, nint=nintx, phase=False), nint=nintx, phase=False))
    bounds_ = bounds__

    coeffs = get_bound_coeffs(func, bounds_, norder, reterr=False).T
    bounds = bounds_[:-1]

    # Round bounds to given significant figures
    coeffs[0] = round_sig(coeffs[0], nsig0)
    coeffs[1] = round_sig(coeffs[1], nsig1)

    nlab = int(np.ceil(np.log2(len(bounds))))

    ###################### Playground ################################

    nint1 = qt.get_nint(coeffs[0])
    nint2 = nintx + nint1
    nint3 = qt.get_nint(coeffs[1])

    npres1 = qt.get_npres(coeffs[0])
    npres2 = (nx - nintx) + npres1
    npres3 = qt.get_npres(coeffs[1])

    n1 = npres1 + nint1 + 1
    n2 = npres2 + nint2 + 1
    n3 = npres3 + nint3 + 1

    ########### round gradients #######################

    rcoeffs = []
    for coeff in coeffs[0]:
        bitstr = qt.my_binary_repr(coeff, 100, nint=nint1, phase=True)
        if bitstr[ncut0]=='0':
            rem = 0.
        else:
            rem = 2**(-(ncut0-nint1-1))
        if bitstr[0]=='1':
            rem = rem*-1
        rcoeff1 = qt.bin_to_dec(bitstr[:ncut0], nint=nint1, phase=True)+rem
        rcoeff2 = qt.bin_to_dec(bitstr[:ncut0], nint=nint1, phase=True)
        rcoeff = np.array([rcoeff1,rcoeff2])[np.argmin(np.abs([rcoeff1-coeff,rcoeff2-coeff]))]
        rcoeffs.append(rcoeff)
    rcoeffs = np.array(rcoeffs)
    coeffs[0] = rcoeffs

    fdifs = func(xs) - qt.piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds_)
    coeffs_ = []
    bounds__ = bounds_
    bounds__[-1] = np.inf
    for i in np.arange(len(bounds__))[:-1]:
        coeffs_.append(np.mean(fdifs[np.greater_equal(xs,bounds__[i])&np.greater(bounds__[i+1],xs)]))
    coeffs[1] = np.array(coeffs_)
    coeffs[1] = round_sig(coeffs[1], nsig1)
    nint3 = qt.get_nint(coeffs[1])
    npres3 = qt.get_npres(coeffs[1])
    n3 = npres3 + nint3 + 1

    rcoeffs = []
    for coeff in coeffs[1]:
        bitstr = qt.my_binary_repr(coeff, 100, nint=nint3, phase=True)
        if bitstr[ncut1]=='0':
            rem = 0.
        else:
            rem = 2**(-(ncut1-nint3-1))
        if bitstr[0]=='1':
            rem = rem*-1
        rcoeff1 = qt.bin_to_dec(bitstr[:ncut1], nint=nint3, phase=True)+rem
        rcoeff2 = qt.bin_to_dec(bitstr[:ncut1], nint=nint3, phase=True)
        rcoeff = np.array([rcoeff1,rcoeff2])[np.argmin(np.abs([rcoeff1-coeff,rcoeff2-coeff]))]
        rcoeffs.append(rcoeff)
    rcoeffs = np.array(rcoeffs)

    coeffs[1] = rcoeffs

    ############## and repeat ########################

    A1x = qt.piecewise_poly(xs, np.array([coeffs[0],np.zeros(len(coeffs[1]))]).T, bounds_)
    A1x_A0 = qt.piecewise_poly(xs, coeffs.T, bounds_)
    coeffs_old = np.copy(coeffs)

    coeffs[0] = np.array([*coeffs[0,2**(nlab-1)+1:],*coeffs[0,:2**(nlab-1)+1]])
    coeffs[1] = np.array([*coeffs[1,2**(nlab-1)+1:],*coeffs[1,:2**(nlab-1)+1]])

    coeffs[0] = np.array([*coeffs[0,-2:],*coeffs[0,:-2]])
    coeffs[1] = np.array([*coeffs[1,-2:],*coeffs[1,:-2]])

    nint1 = qt.get_nint(coeffs[0])
    nint2 = nintx + nint1# - 1
    nint3 = qt.get_nint(coeffs[1])

    npres1 = ncut0-nint1
    npres2 = (nx - nintx) + npres1
    npres3 = ncut1-nint2

    n1 = npres1 + nint1 + 1
    n2 = npres2 + nint2 + 1
    n3 = npres3 + nint3 + 1

    while np.min(A1x)>qt.bin_to_dec('1'+'0'*(n2-3)+'1', nint=nint2-1, phase=phase) and np.max(A1x)<qt.bin_to_dec('0'+'1'*(n2-2)+'1', nint=nint2-1, phase=phase):
        nint2 = nint2 - 1
        n2 = npres2+nint2+1

    nint2 = nint2 + 1
    n2 = npres2 + nint2 + 1

    n = n2
    nc = n1

    nintcs = np.array([[nint1,nint3]])
    nint = nint2

    if 16*(2**(nc+n+nx+nlab))/2**20>7568:
        raise ValueError('Too many qubits!',nc+n+nx+nlab)

    return n, nc, nlab, nint, nintcs, coeffs, bounds

def plot_sim_phase(statename, plotname, nlab, ncut0, ncut1, nsig0, nsig1, fmin=40., fmax=168., m1=35., m2=30., Tfrac=100., beta=0., sig=0.):
    ######### Establish qubit numbers and precision ###############

    state_v = np.load(statename)
    nx = int(np.log2(len(state_v)))

    ######## Frequency ranges and data parameters ######################

    nintx = int(np.ceil(np.log2((fmax-fmin))))

    xmax = np.power(2,nintx) - np.power(2,nintx-nx)
    xmin = 0.

    def m_geo(m):
        return (4.926e-6)*m

    df = (fmax-fmin)/(2**nx)
    T = 1./df

    ####### Physical system parameters ###################

    m1 = m_geo(m1)
    m2 = m_geo(m2)
    tc = T + (T/Tfrac)
    DT = tc%T
    Mt = m1 + m2
    nu = (m1*m2)/Mt
    eta = nu/Mt
    Mc = Mt*eta**(3./5)

    def x_trans(x):
        x = x/xmax
        x = x*(fmax-fmin-df)
        x = x + fmin
        return x

    xs = np.linspace(xmin,xmax,2**(nx))
    xsT = x_trans(xs)

    def f_x(x, eps1=1, eps2=1, factor=1):
        x = x_trans(x)
        out = ((eps1*(3./128))*((np.pi*Mc*x)**(-5./3))*( 1.+ (20./9)*((743./336)+(11./4)*eta)*(np.pi*Mt*x)**(2./3) -4.*(4.*np.pi - beta)*(np.pi*Mt*x) + 10.*eps2*((3058673./1016064) + (eta*5429./1008) + (617*(eta**2)/144) - sig)*(np.pi*Mt*x)**(4./3)) + 2.*np.pi*x*DT)/(2.*np.pi*factor)
        return out

    def amplitude(nqubit):
        xs = np.linspace(fmin, fmax, 2**nqubit)
        amps = xs**(-7./6)
        norm = np.sqrt(np.sum(np.abs(amps)**2))
        return amps/norm

    n, nc, nlab, nint, nintcs, coeffs, bounds = optimize_coeffs_qubits(f_x, nx, nlab, nintx, ncut0, ncut1, nsig0=nsig0, nsig1=nsig1)

    probs = np.argwhere(np.round(np.abs(state_v)**2,15)>0.)[:,1]

    if probs.shape[0]!=2**nx:
        raise ValueError('The number of non-zero probability states does not correspond to the number of frequency states.')
    
    probs = ((probs/(2**n)) * (2**(nint) + 2**(nint) - 2**(-(n-nint-1)))) - 2**nint

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    ax, ax2 = ax

    ax.tick_params(axis='both', labelsize=ticksize)
    ax.set_ylabel(r'$\Psi(f)$', fontsize=fontsize)

    ax.set_xticks([])
    ax.scatter(xsT, probs, color='black', lw=2)

    for bound in bounds:
        ax.axvline(x_trans(bound), color='black', ls=':', lw=1)

    ax.set_xlim(fmin-1,fmax+1)

    ax.plot(xsT, f_x(xs), color='black', lw=1)

    ax2.set_xlabel(r'$f$ (Hz)', fontsize=fontsize)
    ax2.tick_params(axis='both', labelsize=ticksize)
    ax2.tick_params(axis='y', labelsize=ticksize-2)
    ax2.set_xlim(fmin-1,fmax+1)

    ax2.set_ylabel(r'$\Delta\Psi(f)$', fontsize=fontsize)

    ax2.scatter(xsT, probs-f_x(xs), color='black', lw=2)
    
    fig.tight_layout()

    fig.savefig(plotname, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Plot the simulated state vector compared to target state")
    parser.add_argument('--statename', help="Numpy file with saved state vector.")
    parser.add_argument('--plotname', help="Name of plot file.")
    parser.add_argument('--nlab', help="Number of label qubits.", default=4)
    parser.add_argument('--ncut0', help="Number of bits to round coefficient 0 to.", default=7)
    parser.add_argument('--ncut1', help="Number of bits to round coefficient 1 to.", default=8)
    parser.add_argument('--nsig0', help="Round coefficient 0 to given significant figure.", default=4)
    parser.add_argument('--nsig1', help="Round coefficient 1 to given significant figure.", default=4)
    parser.add_argument('--fmin', help="Minimum frequency.", default=40.)
    parser.add_argument('--fmax', help="Maximum frequency.", default=168.)
    parser.add_argument('--m1', help="Component mass 1.", default=30.)
    parser.add_argument('--m2', help="Component mass 2.", default=35.)
    parser.add_argument('--beta', help="Beta spin parameter.", default=0.)
    parser.add_argument('--sig', help="Sigma spin parameter.", default=0.)
    parser.add_argument('--Tfrac', help=" ", default=100.)

    opt = parser.parse_args()

    plot_sim_phase(opt.statename, opt.plotname, opt.nlab, ncut0=opt.ncut0, ncut1=opt.ncut1, nsig0=opt.nsig0, nsig1=opt.nsig1, fmin=opt.fmin, fmax=opt.fmax, m1=opt.m1, m2=opt.m2, Tfrac=opt.Tfrac, beta=opt.beta, sig=opt.sig)
