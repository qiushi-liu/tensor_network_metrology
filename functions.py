import numpy as np
from opt_einsum import *
import cvxpy as cp
import mosek
import scipy

# define Pauli matrices
Pauli_X = np.array([[0, 1], [1, 0]])
Pauli_Y = np.array([[0, -1j], [1j, 0]])
Pauli_Z = np.array([[1, 0], [0, -1]])

# partial trace over the subsystem at axis for numpy 2d array (assume that the matrix for each subsystem is square)
# rho: a matrix (numppy 2d array))
# dims: a list containing the dimension of each subsystem
# axis: the index of the subsytem to be traced out
def np_partial_trace(rho, dims, axis):
    dims_ = np.array(dims)
    
    # return full trace of the matrix consisting of single subsystem
    if dims_.size == 1:
        return np.trace(rho)
    
    # reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # each subsystem gets one index for its row and another one for its column
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))
    
    # if axis is an integer, trace over single subsystem
    if isinstance(axis, int):
        traced_out_rho = np.trace(reshaped_rho, axis1=axis, axis2=dims_.size+axis)

        # traced_out_rho is still in the shape of a tensor
        # reshape back to a matrix
        dims_untraced = np.delete(dims_, axis)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])
    
    # if axis is a list, trace over multiple subsystems
    elif isinstance(axis, list):
        axis = sorted(axis)
        dims_untraced = dims_
        traced_out_rho = reshaped_rho
        for i in range(len(axis)):
            traced_out_rho = np.trace(traced_out_rho, axis1=axis[i]-i, axis2=dims_untraced.size+axis[i]-i)
            dims_untraced = np.delete(dims_untraced, axis[i]-i)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])
    
# the i-th basis vector of n-dim Hilbert space
def basis_state(n, i):
    return np.array([0]*i + [1] + [0]*(n-1-i)).reshape(n,1)

# unnormalized maximally entangled state vector of two d-dim subsystems: \sum_i \ket{ii}
def max_entangled_state(d):
    state = 0
    for i in range(d):
        state += np.kron(basis_state(d, i), basis_state(d, i))
    return state

# unnormalized maximally entangled state (density matrix) of two d-dim subsystems: \sum_{ij} \ket{ii} \bra{jj}
def max_entangled_dm(d):
    return max_entangled_state(d) @ max_entangled_state(d).conj().T
    
# K_thetas_AD is a list of Kraus operators for a unitary evolution U(theta,t) followed by amplitude damping noise
# theta: the parameter to estimate
# t: the unitary evolution time
# p: the decay parameter
def K_thetas_AD(theta, t, p):
    # U is the unitary evolution encoding theta
    U = np.array([[np.exp(-1j * (theta * t) / 2), 0],
                  [0, np.exp(1j * (theta * t) / 2)]])
    return [np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ U, np.array([[0, np.sqrt(p)], [0, 0]]) @ U]

# dK_thetas_AD is the derivative of K_thetas_AD
def dK_thetas_AD(theta, t, p):
    # dU is the derivative of U
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (theta * t) / 2), 0],
                   [0, (1j * t / 2) * np.exp(1j * (theta * t) / 2)]])
    return [np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ dU, np.array([[0, np.sqrt(p)], [0, 0]]) @ dU]

# K_thetas_AD_noise_signal is a list of Kraus operators for a unitary evolution U(theta,t) following amplitude damping noise
# theta: the parameter to estimate
# t: the unitary evolution time
# p: the decay parameter
def K_thetas_AD_noise_signal(theta, t, p):
    # U is the unitary evolution encoding theta
    U = np.array([[np.exp(-1j * (theta * t) / 2), 0],
                  [0, np.exp(1j * (theta * t) / 2)]])
    return [U @ np.array([[1, 0], [0, np.sqrt(1 - p)]]), U @ np.array([[0, np.sqrt(p)], [0, 0]])]

# dK_thetas_AD is the derivative of K_thetas_AD
def dK_thetas_AD_noise_signal(theta, t, p):
    # dU is the derivative of U
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (theta * t) / 2), 0],
                   [0, (1j * t / 2) * np.exp(1j * (theta * t) / 2)]])
    return [dU @ np.array([[1, 0], [0, np.sqrt(1 - p)]]), dU @ np.array([[0, np.sqrt(p)], [0, 0]])]

# K_thetas_dep is a list of Kraus operators for a unitary evolution U(theta,t) followed by (possibly asymmetric) depolarizing noise
# theta: the parameter to estimate
# t: the unitary evolution time
# px, py, pz: error probability for Pauli X, Y, Z noise
def K_thetas_dep(theta, t, p_x, p_y, p_z):
    # U is the unitary evolution encoding theta
    U = np.array([[np.exp(-1j * (theta * t) / 2), 0],
                  [0, np.exp(1j * (theta * t) / 2)]])
    return [np.sqrt(1-p_x-p_y-p_z) * np.eye(2) @ U, np.sqrt(p_x) * np.array([[0,1], [1,0]]) @ U, 
            np.sqrt(p_y) * np.array([[0,-1j], [1j,0]]) @ U, np.sqrt(p_z) * np.array([[1,0], [0,-1]]) @ U]

# dK_thetas_dep is the derivative of K_thetas_dep
def dK_thetas_dep(theta, t, p_x, p_y, p_z):
    # dU is the derivative of U
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (theta * t) / 2), 0],
                   [0, (1j * t / 2) * np.exp(1j * (theta * t) / 2)]])
    return [np.sqrt(1-p_x-p_y-p_z) * np.eye(2) @ dU, np.sqrt(p_x) * np.array([[0,1], [1,0]]) @ dU, 
            np.sqrt(p_y) * np.array([[0,-1j], [1j,0]]) @ dU, np.sqrt(p_z) * np.array([[1,0], [0,-1]]) @ dU]

# K_thetas_dep_noise_signal is a list of Kraus operators for a unitary evolution U(theta,t) following (possibly asymmetric) depolarizing noise
# theta: the parameter to estimate
# t: the unitary evolution time
# px, py, pz: error probability for Pauli X, Y, Z noise
def K_thetas_dep_noise_signal(theta, t, p_x, p_y, p_z):
    # U is the unitary evolution encoding theta
    U = np.array([[np.exp(-1j * (theta * t) / 2), 0],
                  [0, np.exp(1j * (theta * t) / 2)]])
    return [np.sqrt(1-p_x-p_y-p_z) * U @ np.eye(2), np.sqrt(p_x) * U @ np.array([[0,1], [1,0]]), 
            np.sqrt(p_y) * U @ np.array([[0,-1j], [1j,0]]), np.sqrt(p_z) * U @ np.array([[1,0], [0,-1]])]

# dK_thetas_dep is the derivative of K_thetas_dep_noise_signal
def dK_thetas_dep_noise_signal(theta, t, p_x, p_y, p_z):
    # dU is the derivative of U
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (theta * t) / 2), 0],
                   [0, (1j * t / 2) * np.exp(1j * (theta * t) / 2)]])
    return [np.sqrt(1-p_x-p_y-p_z) * dU @ np.eye(2), np.sqrt(p_x) * dU @ np.array([[0,1], [1,0]]), 
            np.sqrt(p_y) * dU @ np.array([[0,-1j], [1j,0]]), np.sqrt(p_z) * dU @ np.array([[1,0], [0,-1]])]

# Choi operator of the channel desribed by Kraus operators K_thetas
def E_theta(K_thetas):
    d = K_thetas[0].shape[1]
    Choi = 0
    for i in range(len(K_thetas)):
        Choi += np.kron(K_thetas[i], np.eye(d)) @ max_entangled_dm(d) @ np.kron(K_thetas[i].conj().T, np.eye(d))
    return Choi

# the derivative of Choi operator of the channel desribed by Kraus operators  K_thetas and their derivatives dK_thetas
def dE_theta(K_thetas, dK_thetas):
    d = K_thetas[0].shape[1]
    dChoi = 0
    for i in range(len(K_thetas)):
        dChoi += (np.kron(dK_thetas[i], np.eye(d)) @ max_entangled_dm(d) @ np.kron(K_thetas[i].conj().T, np.eye(d)) 
        + np.kron(K_thetas[i], np.eye(d)) @ max_entangled_dm(d) @ np.kron(dK_thetas[i].conj().T, np.eye(d)))
    return dChoi