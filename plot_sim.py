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

def plot_sim(statename, plotname, label, fmin=40., fmax=168., m1=35., m2=30., Tfrac=100., beta=0., sig=0.):

    state_v = np.load(statename)

    nx = int(np.log2(len(state_v)))

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
    beta = beta
    sig = sig

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

    target_state = amplitude(nx)

    if statename.split('/')[-1].split('_')[0]=='full':
        target_state = amplitude(nx)*np.exp(2*1.j*np.pi*f_x(np.linspace(xmin,xmax,2**(nx))))

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})

    color1 = 'black'
    color2 = 'grey'

    if np.sum(np.round(state_v.imag,14))!=0.:
        ax[0].scatter(xsT, state_v.imag, color=color2, lw=2)
        ax[0].plot(xsT, target_state.imag, color=color2, ls='-', lw=1)
        ax[1].scatter(xsT, state_v.imag-target_state.imag, color=color2, lw=2)

    ax[0].scatter(xsT, state_v.real, color=color1, lw=2)

    ax[0].plot(xsT, target_state.real, color=color1, ls='-', lw=1)

    ax[0].tick_params(axis='both', labelsize=ticksize)
    ax[1].set_xlabel(r'$f$ (Hz)', fontsize=fontsize)
    ax[0].set_ylabel(r'$'+label+'$', fontsize=fontsize)
    ax[0].set_xlim(fmin-1,fmax+1)

    ax[0].set_xticks([])

    ax[1].scatter(xsT, state_v.real-target_state.real, color=color1, lw=2)
    ax[1].set_ylabel(r'$\Delta '+label+'$', fontsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=ticksize)
    ax[1].set_xlim(fmin-1,fmax+1)

    ax[1].set_ylim(-0.05, 0.05)

    fidelity = np.abs(np.dot(target_state,np.conjugate(state_v)))**2
    fig.tight_layout()

    print('Fidelity:',fidelity,'Mismatch:', 1-np.sqrt(fidelity))

    fig.savefig(plotname, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Plot the simulated state vector compared to target state")
    parser.add_argument('--statename', help="Numpy file with saved state vector.")
    parser.add_argument('--plotname', help="Name of plot file.")
    parser.add_argument('--label', help="y-axis label name (in math mode).")
    parser.add_argument('--fmin', help="Minimum frequency.", default=40.)
    parser.add_argument('--fmax', help="Maximum frequency.", default=168.)
    parser.add_argument('--m1', help="Component mass 1.", default=30.)
    parser.add_argument('--m2', help="Component mass 2.", default=35.)
    parser.add_argument('--beta', help="Beta spin parameter.", default=0.)
    parser.add_argument('--sig', help="Sigma spin parameter.", default=0.)
    parser.add_argument('--Tfrac', help=" ", default=100.)

    opt = parser.parse_args()

    plot_sim(opt.statename, opt.plotname, opt.label, fmin=opt.fmin, fmax=opt.fmax, m1=opt.m1, m2=opt.m2, Tfrac=opt.Tfrac, beta=opt.beta, sig=opt.sig)
