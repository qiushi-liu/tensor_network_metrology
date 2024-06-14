# Tensor network algorithm for optimizing control-enhanced quantum metrology
This repository contains the Python code accompanying the article "Efficient tensor networks for control-enhanced quantum metrology"([arXiv:2403.09519](https://arxiv.org/abs/2403.09519)). We implement the tensor network algorithm to efficiently maximize the quantum Fisher information (QFI) of the output state obtained by several types of control strategies. Our approach covers a general and practical scenario where the experimenter applies $N−1$ interleaved control operations between $N$ queries of the channel to estimate and uses no or bounded ancilla. 
## Requirements
The code for tensor contraction requires the Python package [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/). The code for SDP requires the Python package [CVXPY](https://www.cvxpy.org) with the optimizer [MOSEK](https://www.mosek.com). The code for variational circuit optimization requires the Python package [PennyLane](https://pennylane.ai/).
## Description
* `functions.py` contains useful functions for the setup.
* `arbitrary_control.py` constructs the problem of QFI optimization with arbitrary completely positive trace-preserving (CPTP) control operations.
* `identical_control.py` constructs the problem of QFI optimization with identical CPTP control operations.
* `unitary_control.py` constructs the problem of QFI optimization with variational unitary control operations.
* `identical_unitary_control.py` constructs the problem of QFI optimization with identical variational unitary control operations.
* `solve.py` provides two examples for applying the optimization algorithm.
## How to use
* We have provided two examples in `solve.py`, for (1) phase estimation with the amplitude damping noise (identical control, 0 ancilla) and (2) phase estimation with the bit flip noise (arbitrary control, 1 ancilla).
* To adjust the setup, feel free to choose one of these types of the control strategy in `solve.py`:
```python
control_type = 'ac' # arbitrary control
```
```python
control_type = 'ic' # identical control
```
```python
control_type = 'uc' # unitary control
```
```python
control_type = 'iuc' # identical unitary control
```
and assign the ancilla dimension: for example the following setup
```python
d = 2 # system dimension
d_a = 1 # ancilla dimension
d_tot = d*d_a # total dimemsion
n_qubits = 1 # number of qubits for variational control
num_layers = 1 # number of layers for variational control
```
uses no ancilla. The initialization of the input state and control can also be adjusted, similar to
```python
# initial input state
psi_0 = np.sqrt(1/2)*basis_state(2, 0) + np.sqrt(1/2)*basis_state(2, 1) 
rho_0 = psi_0 @ psi_0.conj().T

# initial control
controls_0 = [np.reshape(max_entangled_dm(d_tot), (d, d_a, d, d_a, d, d_a, d, d_a))] # arbitrary control
control_0 = np.reshape(max_entangled_dm(d_tot), (d, d_a, d, d_a, d, d_a, d, d_a)) # identical control
np.random.seed(0)
all_control_parameters_0 = qml.numpy.random.uniform(
                    0, 2*np.pi, (N_steps_min-1, num_layers, n_qubits, 3), requires_grad=True
        ) # unitary control
control_parameters_0 = qml.numpy.random.uniform(
                    0, 2*np.pi, (num_layers, n_qubits, 3), requires_grad=True
        ) # identical unitary control
```
By assigning values to all the necessary parameters, call the following function to solve the QFI for the number of queries from `N_steps_min` to `N_steps_max`:
```python
solve_QFIs(control_type, N_steps_min, N_steps_max, rho, control, E_theta_AD, dE_theta_AD, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0)
```
The other noise models can also be used for QFI evaluation, by defining the Kraus operators with the derivative in `functions.py`, similar to
```python
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
```
and compute the Choi operators with the derivative, similar to
```python
# amplitude damping noise
K_thetas_AD = K_thetas_AD_noise_signal(theta, t, p)
dK_thetas_AD = dK_thetas_AD_noise_signal(theta, t, p)
E_theta_AD = E_theta(K_thetas_AD)
dE_theta_AD = dE_theta(K_thetas_AD, dK_thetas_AD)
```
