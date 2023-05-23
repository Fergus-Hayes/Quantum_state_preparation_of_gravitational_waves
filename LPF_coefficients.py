# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This Python script implements Remez's algorithm (https://en.wikipedia.org/wiki/Remez_algorithm).
# To use the script, you can modify what happens when this script is executed:
# See the `if __name__ == "__main__":` branch near the end of this file.
#
# Alternatively, `run_remez` can be imported and run directly, see the docstring
# of run_remez.

import numpy as np
from matplotlib import pyplot as plt
import qiskit_tools as qt

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

def get_bound_coeffs(func, bounds, norder, reterr=False):
    if np.array(bounds).ndim==0:
        print('Bounds must be a list of two entries!')
    if len(bounds)==1:
        print('Bounds must be a list of two entries!')
        
    coeffs = []
    errs = []
    for i in np.arange(len(bounds))[:-1]:
        coeffs_, err_ = run_remez(func, bounds[i], bounds[i+1], norder)
        coeffs.append(np.array(coeffs_))
        errs.append(err_)
    if reterr:
        return np.array(coeffs), np.array(errs)
    else:
        return np.array(coeffs)

# Return n chebyshev nodes on the interval (a,b)
def _get_chebyshev_nodes(n, a, b):
    nodes = [.5 * (a + b) + .5 * (b - a) * np.cos((2 * k + 1) / (2. * n) * np.pi)
             for k in range(n)]
    return nodes

# Return the error on given nodes of a polynomial with coefficients poly_coeff
# approximating the function with function values exact_values (on these nodes).
def _get_errors(exact_values, poly_coeff, nodes):
    ys = np.polyval(poly_coeff, nodes)
    for i in range(len(ys)):
        ys[i] = abs(ys[i] - exact_values[i])
    return ys

"""
Return the coefficients of a polynomial of degree d approximating
the function fun on the interval (a,b).
Args:
    fun: Function to approximate
    a: Left interval border
    b: Right interval border
    d: The polynomial degree will be d, 2*d or 2*d + 1 depending
        on the values of odd and even below
    odd: If True, use odd polynomial of degree 2*d+1
    even: If True, use even polynomial of degree 2*d
    tol: Tolerance to use when checking for convergence
Returns: Tuple where the first entry is the achieved absolute error
    and the second entry is a list of the polynomial coefficients in
    the order that is required by the QDK Numerics library. This is
    the inverse order compared to what np.polyval expects.
"""
def run_remez(fun, a, b, d=5, odd=False, even=False, tol=1.e-13):
    finished = False
    # initial set of points for the interpolation
    cn = _get_chebyshev_nodes(d + 2, a, b)
    # mesh on which we'll evaluate the error
    cn2 = _get_chebyshev_nodes(100 * d, a, b)    
    # do at most 50 iterations and cancel if we "lose" an interpolation
    # point
    it = 0
    while not finished and len(cn) == d + 2 and it < 50:
        it += 1
        # set up the linear system of equations for Remez' algorithm
        b = np.array([fun(c) for c in cn])
        A = np.matrix(np.zeros([d + 2,d + 2]))
        for i in range(d + 2):
            x = 1.
            if odd:
                x *= cn[i]
            for j in range(d + 2):
                A[i, j] = x
                x *= cn[i]
                if odd or even:
                    x *= cn[i]
            A[i, -1] = (-1)**(i + 1)
        # this will give us a polynomial interpolation
        res = np.linalg.solve(A, b)

        # add padding for even/odd polynomials
        revlist = reversed(res[0:-1])
        sc_coeff = []
        for c in revlist:
            sc_coeff.append(c)
            if odd or even:
                sc_coeff.append(0)
        if even:
            sc_coeff = sc_coeff[0:-1]
        # evaluate the approximation error
        errs = _get_errors([fun(c) for c in cn2], sc_coeff, cn2)
        maximum_indices = []

        # determine points of locally maximal absolute error
        if errs[0] > errs[1]:
            maximum_indices.append(0)
        for i in range(1, len(errs) - 1):
            if errs[i] > errs[i-1] and errs[i] > errs[i+1]:
                maximum_indices.append(i)
        if errs[-1] > errs[-2]:
            maximum_indices.append(-1)

        # and choose those as new interpolation points
        # if not converged already.
        finished = True
        for idx in maximum_indices[1:]:
            if abs(errs[idx] - errs[maximum_indices[0]]) > tol:
                finished = False

        cn = [cn2[i] for i in maximum_indices]

    return sc_coeff, max(abs(errs))


if __name__ == "__main__":
    # the function to approximate
    def f(x):
       return np.sin(x)

    # f(x) is an odd function, so we can approximate it
    # using a polynomial of degree 2n+1
    odd = True
    even = False

    # approximate f(x) on the interval (a,b), where
    a = 0.
    b = np.pi

    # set the polynomial degree. If the function is even or odd, the polynomial
    # will be of degree `2*degree` or `2*degree + 1`, respectively.
    degree = 3

    # run Remez' algorithm
    err, coeffs = run_remez(f, a, b, degree, odd, even)

    # and output the coefficients & achieved approximation error
    oddEvenStr = ""
    if odd:
        oddEvenStr = " for odd powers of x"
    if even:
        oddEvenStr = " for even powers of x"

    print("Coefficients{}: {}".format(oddEvenStr, list(reversed(coeffs))))
    print("The polynomial achieves an L_inf error of {}.".format(err))
