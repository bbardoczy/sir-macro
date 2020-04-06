import numpy as np
import scipy.linalg
import inspect


'''Part 1: Manipulating Jacobians'''


def pack_vectors(vs, names, T):
    """Dictionary of vectors into a single vector."""
    v = np.zeros(len(names)*T)
    for i, name in enumerate(names):
        if name in vs:
            v[i*T:(i+1)*T] = vs[name]
    return v


def unpack_vectors(v, names, T):
    """Single vector to dictionary of vectors."""
    vs = {}
    for i, name in enumerate(names):
        vs[name] = v[i*T:(i+1)*T]
    return vs


def pack_jacobians(jacdict, inputs, outputs, T):
    """If we have T*T jacobians from nI inputs to nO outputs in jacdict, combine into (nO*T)*(nI*T) jacobian matrix."""
    nI, nO = len(inputs), len(outputs)

    outjac = np.empty((nO * T, nI * T))
    for iO in range(nO):
        subdict = jacdict.get(outputs[iO], {})
        for iI in range(nI):
            outjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = subdict.get(inputs[iI], np.zeros((T, T)))
    return outjac


def J_to_HU(J, H, unknowns, targets):
    """Jacdict to LU-factored jacobian."""
    H_U = pack_jacobians(J, unknowns, targets, H)
    H_U_factored = factor(H_U)
    return H_U_factored


'''Part 2: Efficient Newton step'''


def factor(X):
    return scipy.linalg.lu_factor(X)


def factored_solve(Z, y):
    return scipy.linalg.lu_solve(Z, y)


'''Part 3: Convenience'''


def input_list(f):
    """Return list of function inputs"""
    return inspect.getfullargspec(f).args
