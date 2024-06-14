from arbitrary_control import *
from identical_control import *
from unitary_control import *
from identical_unitary_control import *

# solve the QFI with different control strategies for different numbers of queries to E_theta, from N_steps_min to N_steps_max
# control_type: string, 'ac' for arbitrary control, 'ic' for identical control, 'uc' for unitary control, 'iuc' for identical unitary control
# N_steps_min: smallest number of E_theta (>=2)
# N_steps_max: largest number of E_theta 
# rho: initial input state
# control: Choi operators of initial control operations or parameters determining the unitary control
# E_theta: Choi operator of the channel to estimate
# dE_theta: derivative of E_theta
# d: system dimension 
# d_a: ancilla dimension
# n_qubits: number of qubits for the variational circuit
# num_layers: number of layers for the variational circuit
# iterations: maximal number of iterations
# eps_SLD: numerical tolerance in rounding the nonzero denominator for computing the SLD
# eps_QFI: numerical tolerance for increase of QFI
# decay_parameter0: initial perturbation noise strength (adding an additional depolarizing noise for better convergence)
def solve_QFIs(control_type, N_steps_min, N_steps_max, rho, control, E_theta, dE_theta, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0):
    QFIs = []
    if control_type == 'ac': # arbitrary control
        for N_steps in range(N_steps_min, N_steps_max+1):
            print('N_steps', N_steps)
            QFI = sequential_QFI_arbitrary_control(rho, control, E_theta, dE_theta, N_steps, d, d_a, iterations, eps_SLD, eps_QFI, decay_parameter0)
            QFIs.append(QFI[0])
            rho = QFI[1]
            control = QFI[2] + [np.reshape(max_entangled_dm(d_tot), (d, d_a, d, d_a, d, d_a, d, d_a))]
    if control_type == 'ic': # identical control
        for N_steps in range(N_steps_min, N_steps_max+1):
            print('N_steps', N_steps)
            QFI = sequential_QFI_same_control(rho, control, E_theta, dE_theta, N_steps, d, d_a, iterations, eps_SLD, eps_QFI, decay_parameter0)
            QFIs.append(QFI[0])
            rho = QFI[1]
            control = QFI[2]
    if control_type == 'uc': # unitary control
        for N_steps in range(N_steps_min, N_steps_max+1):
            print('N_steps', N_steps)
            QFI = sequential_QFI_unitary_control(rho, control, E_theta, dE_theta, N_steps, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0)
            QFIs.append(QFI[0])
            rho = QFI[1]
            control = qml.numpy.concatenate((QFI[2], qml.numpy.random.uniform(
                    0, 2*np.pi, (1, num_layers, n_qubits, 3), requires_grad=True
        )))
    if control_type == 'iuc': # identical unitary control
        for N_steps in range(N_steps_min, N_steps_max+1):
            print('N_steps', N_steps)
            QFI = sequential_QFI_same_unitary_control(rho, control, E_theta, dE_theta, N_steps, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0)
            QFIs.append(QFI[0])
            rho = QFI[1]
            control = QFI[2]
    return QFIs

N_steps_max = 10
N_steps_min = 2
theta = 1.0
t = 1.0
p_x = 0.1 # bit flip noise
p_y = 0
p_z = 0
p = 0.1 # AD noise
iterations = 200

# bit flip noise
K_thetas_dep = K_thetas_dep_noise_signal(theta, t, p_x, p_y, p_z)
dK_thetas_dep = dK_thetas_dep_noise_signal(theta, t, p_x, p_y, p_z)
E_theta_BF = E_theta(K_thetas_dep)
dE_theta_BF = dE_theta(K_thetas_dep, dK_thetas_dep)

# amplitude damping noise
K_thetas_AD = K_thetas_AD_noise_signal(theta, t, p)
dK_thetas_AD = dK_thetas_AD_noise_signal(theta, t, p)
E_theta_AD = E_theta(K_thetas_AD)
dE_theta_AD = dE_theta(K_thetas_AD, dK_thetas_AD)

# example: amplitude damping noise, identical control, 0 ancilla
print('******amplitude damping noise, identical control, 0 ancilla******')
control_type = 'ic' # identical control
d = 2 # system dimension
d_a = 1 # ancilla dimension
d_tot = d*d_a # total dimemsion
n_qubits = 1 # number of qubits for variational control
num_layers = 1 # number of layers for variational control

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

rho = rho_0
if control_type == 'ac':
    control = controls_0 
if control_type == 'ic':
    control = control_0 
if control_type == 'uc':
    control = all_control_parameters_0
if control_type == 'iuc':
    control = control_parameters_0
eps_SLD = 1e-10
eps_QFI = 1e-12
decay_parameter0 = 0.01
print('QFI (amplitude damping noise, identical control, 0 ancilla): ', solve_QFIs(control_type, N_steps_min, N_steps_max, rho, control, E_theta_AD, dE_theta_AD, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0))


# example: bit flip noise, arbitrary control, 1 ancilla
print('******bit flip noise, arbitrary control, 1 ancilla******')
control_type = 'ac' # arbitrary control
d = 2 # system dimension
d_a = 2 # ancilla dimension
d_tot = d*d_a # total dimemsion
n_qubits = 2 # number of qubits for variational control
num_layers = 3 # number of layers for variational control

# initial input state
psi_0 = np.sqrt(1/2)*np.kron(basis_state(2, 0), basis_state(2, 0)) + np.sqrt(1/2)*np.kron(basis_state(2, 1), basis_state(2, 1))
rho_0 = psi_0 @ psi_0.conj().T

# initial control
controls_0 = [np.reshape(max_entangled_dm(d_tot), (d, d_a, d, d_a, d, d_a, d, d_a))] # arbitrary control
control_0 = np.reshape(max_entangled_dm(d_tot), (d, d_a, d, d_a, d, d_a, d, d_a)) # identical control
np.random.seed(0)
all_control_parameters_0 = qml.numpy.random.uniform(
                    0, 2*np.pi, (1, num_layers, n_qubits, 3), requires_grad=True
        ) # unitary control
control_parameters_0 = qml.numpy.random.uniform(
                    0, 2*np.pi, (num_layers, n_qubits, 3), requires_grad=True
        ) # identical unitary control

rho = rho_0
if control_type == 'ac':
    control = controls_0 
if control_type == 'ic':
    control = control_0 
if control_type == 'uc':
    control = all_control_parameters_0
if control_type == 'iuc':
    control = control_parameters_0
eps_SLD = 1e-10
eps_QFI = 1e-12
decay_parameter0 = 0.1
print('QFI (bit flip noise, arbitrary control, 1 ancilla): ', solve_QFIs(control_type, N_steps_min, N_steps_max, rho, control, E_theta_BF, dE_theta_BF, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0))
