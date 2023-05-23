from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
import qiskit_tools as qt
from LPF_coefficients import get_bound_coeffs, optimize_coeffs_qubits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib, argparse
import plot_sim as plsim

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
width=0.75
color='black'
fontsize=28
ticksize=22
figsize=(10,10)

def sim_Af_PQC(nx, nlab, ncut0, ncut1, nsig0, nsig1, statename='./state_vectors/psi_f', fmin=40., fmax=168., m1=35., m2=30., Tfrac=100., beta=0., sig=0.):

    phase = True

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

    def amplitude(nqubit):
        xs = np.linspace(fmin, fmax, 2**nqubit)
        amps = xs**(-7./6)
        norm = np.sqrt(np.sum(np.abs(amps)**2))
        return amps/norm

    def f_x(x, eps1=1, eps2=1, factor=1):
        x = x_trans(x)
        out = ((eps1*(3./128))*((np.pi*Mc*x)**(-5./3))*( 1.+ (20./9)*((743./336)+(11./4)*eta)*(np.pi*Mt*x)**(2./3) -4.*(4.*np.pi - beta)*(np.pi*Mt*x) + 10.*eps2*((3058673./1016064) + (eta*5429./1008) + (617*(eta**2)/144) - sig)*(np.pi*Mt*x)**(4./3)) + 2.*np.pi*x*DT)/(2.*np.pi*factor)
        return out

    n, nc, nlab, nint, nintcs, coeffs, bounds = optimize_coeffs_qubits(f_x, nx, nlab, nintx, ncut0, ncut1, nsig0=nsig0, nsig1=nsig1)

    print('Qubits:', nx, n, nc, nlab)
    print('Integer qubits:', nintx, nint, nintcs[0,0], nintcs[0,1])
    print('Memory:', 16*(2**(nc+n+nx+nlab))/2**20)

    if 16*(2**(nc+n+nx+nlab))/2**20>7568:
        raise ValueError('Too many qubits!',nc+n+nx+nlab)

    #####################################################################

    q_x = QuantumRegister(nx, 'q_x')
    q_y = QuantumRegister(n, 'q_y')
    q_lab = QuantumRegister(nlab, 'q_lab')
    q_coff = QuantumRegister(nc, 'q_coff')

    circ = QuantumCircuit(q_x, q_y, q_lab, q_coff)

    weights = np.load('./weights/weights_20_2260.npy')
    PQC = qt.PQC_realamp(circ, q_x, weights, wrap=True)
    circ.append(PQC, q_x);

    if list(dict(circ.decompose(reps=10).count_ops()).keys())!=['u', 'cx']:
        raise ValueError('Cannot decompose circuit into u and CX gates.')

    print('CNOT gate count:',dict(circ.decompose(reps=10).count_ops())['cx'])
    
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ, backend)
    result = job.result()
    state_vector = result.get_statevector()

    state_vector = np.asarray(state_vector).reshape((2**nc,2**nlab,2**n,2**nx))
    state_v = state_vector[0,0,0,:].flatten()
    print(np.sum(np.abs(state_v)**2))

    np.save(statename, state_v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Plot the simulated state vector compared to target state")
    parser.add_argument('--statename', help="Numpy file with saved state vector.", default='./state_vectors/amp_state_QGAN')
    parser.add_argument('--nx', help="Number of frequency qubits.", default=6)
    parser.add_argument('--nlab', help="Number of label qubits.", default=4)
    parser.add_argument('--ncut0', help="Number of bits to round coefficient 0 to.", default=7)
    parser.add_argument('--ncut1', help="Number of bits to round coefficient 1 to.", default=64)
    parser.add_argument('--nsig0', help="Round coefficient 0 to given significant figure.", default=10)
    parser.add_argument('--nsig1', help="Round coefficient 1 to given significant figure.", default=10)
    parser.add_argument('--fmin', help="Minimum frequency.", default=40.)
    parser.add_argument('--fmax', help="Maximum frequency.", default=168.)
    parser.add_argument('--m1', help="Component mass 1.", default=30.)
    parser.add_argument('--m2', help="Component mass 2.", default=35.)
    parser.add_argument('--beta', help="Beta spin parameter.", default=0.)
    parser.add_argument('--sig', help="Sigma spin parameter.", default=0.)
    parser.add_argument('--Tfrac', help=" ", default=100.)

    opt = parser.parse_args()

    sim_Af_PQC(opt.nx, opt.nlab, statename=opt.statename, ncut0=opt.ncut0, ncut1=opt.ncut1, nsig0=opt.nsig0, nsig1=opt.nsig1, fmin=opt.fmin, fmax=opt.fmax, m1=opt.m1, m2=opt.m2, Tfrac=opt.Tfrac, beta=opt.beta, sig=opt.sig)
    
    plsim.plot_sim(opt.statename+'.npy', './figures/Q_GAN_amp_state.png', r'\tilde{A}(f)', fmin=opt.fmin, fmax=opt.fmax, m1=opt.m1, m2=opt.m2, Tfrac=opt.Tfrac, beta=opt.beta, sig=opt.sig)
