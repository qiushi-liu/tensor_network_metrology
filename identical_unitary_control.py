# tensor network optimization for identical variational unitary control operations

from functions import *
import pennylane as qml

# compute the gradients of the current objective 2*f_1-f_2 (for computing QFI) with respect to variational parameters
# variational_parameters: variational parameters for onr control operation
# variational_unitary: function of variational parameters, store the real and complex parts as an additional axis of the array
# E_theta: Choi operator of the channel to estimate
# dE_tensor_network: tensor network for computing the derivative of E_theta
# rho: current input state
# SLD_operator: current operator X
# d: system dimension 
# d_a: ancilla dimension
# N_steps: number of channels to estimate
def grad_QFI_fn(variational_parameters, variational_unitary, E_theta, dE_tensor_network, rho, SLD_operator, d, d_a, N_steps):
    d_tot = d*d_a
    control_unitary = variational_unitary(variational_parameters)
    control_unitary_complex = control_unitary[0] + 1j*control_unitary[1]
    control_Choi_vec = qml.numpy.reshape(control_unitary_complex, (d_tot**2, 1))
    control_Choi = control_Choi_vec @ control_Choi_vec.conj().T
    reshaped_control = np.reshape(control_Choi, (d, d_a, d, d_a, d, d_a, d, d_a))
    controls = [reshaped_control]*(N_steps-1)

    # compute the Jacobian
    shape_parameters = variational_parameters.shape 
    shape_unitary = control_unitary.shape #(2, d_tot, d_tot)
    shape_jac_unitary_parameters = shape_unitary + shape_parameters
    jac_unitary_parameters = qml.jacobian(variational_unitary)(variational_parameters)
    jac_unitary_parameters = np.reshape(jac_unitary_parameters, (2, d_tot**2, np.prod(shape_parameters)))
    jac_unitary_parameters_re = jac_unitary_parameters[0]
    jac_unitary_parameters_im = jac_unitary_parameters[1]
    jac_unitary_parameters_complex = jac_unitary_parameters_re + 1j*jac_unitary_parameters_im
    
    reshaped_control_unitary = np.reshape(control_unitary, (2, d_tot**2))
    variational_unitary_re = reshaped_control_unitary[0]
    variational_unitary_im = reshaped_control_unitary[1]
    variational_unitary_complex = variational_unitary_re + 1j*variational_unitary_im
    grad_QFI_parameters = np.zeros_like(variational_parameters)
    
    total_X = 0
    end_tensor_2 = contract(np.reshape(-SLD_operator @ SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)), 
                            E_theta, ('i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1)), 
                            ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)))
    end_tensor_1 = contract(np.reshape(2*SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)), 
                        dE_tensor_network[0], ('k1', 'i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1)),
                        ('k1', 'i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)))
    
    later_circuit_tensors_2 = [end_tensor_2]
    later_circuit_tensors_1 = [end_tensor_1]
    for i in range(N_steps-2):
        end_tensor_2 = contract(end_tensor_2, ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)), 
                    controls[-i-1], ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1)),
                                 E_theta, ('i{}'.format(2*N_steps-2), 'i{}'.format(2*N_steps-3), 'j{}'.format(2*N_steps-2), 'j{}'.format(2*N_steps-3)),
                                ('i{}'.format(2*N_steps-3), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-3), 'ja{}'.format(N_steps-1)))
        later_circuit_tensors_2.append(end_tensor_2)
        end_tensor_1 = contract(end_tensor_1, ('k1', 'i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)), 
                    controls[-i-1], ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1)),
                                 dE_tensor_network[2], ('k1', 'k2', 'i{}'.format(2*N_steps-2), 'i{}'.format(2*N_steps-3), 'j{}'.format(2*N_steps-2), 'j{}'.format(2*N_steps-3)),
                                ('k2', 'i{}'.format(2*N_steps-3), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-3), 'ja{}'.format(N_steps-1)))
        later_circuit_tensors_1.append(end_tensor_1)

    start_tensor_2 = contract(E_theta, ('i2', 'i1', 'j2', 'j1'), rho, ('i1', 'ia1', 'j1', 'ja1'),
                          ('i2', 'ia1', 'j2', 'ja1'))

    start_tensor_1 = contract(dE_tensor_network[-2], ('k', 'i2', 'i1', 'j2', 'j1'), rho, ('i1', 'ia1', 'j1', 'ja1'),
                          ('k', 'i2', 'ia1', 'j2', 'ja1'))
    
    for control_index in range(N_steps-1):
        end_tensor_2 = later_circuit_tensors_2[-control_index-1]
        end_tensor_1 = later_circuit_tensors_1[-control_index-1]
        if control_index != 0:
            start_tensor_2 = contract(E_theta, ('i4', 'i3', 'j4', 'j3'),
                controls[control_index-1], ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'),
                start_tensor_2, ('i2', 'ia1', 'j2', 'ja1'), ('i4', 'ia2', 'j4', 'ja2'))
            start_tensor_1 = contract(dE_tensor_network[2], ('k1', 'k2', 'i4', 'i3', 'j4', 'j3'),
                controls[control_index-1], ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'),
                start_tensor_1, ('k2', 'i2', 'ia1', 'j2', 'ja1'), ('k1', 'i4', 'ia2', 'j4', 'ja2'))
        X_second_term = contract(end_tensor_2, ('i3', 'ia2', 'j3', 'ja2'),
                         start_tensor_2, ('i2', 'ia1', 'j2', 'ja1'),
                         ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'))
        X_first_term = contract(end_tensor_1, ('k', 'i3', 'ia2', 'j3', 'ja2'),
                         start_tensor_1, ('k', 'i2', 'ia1', 'j2', 'ja1'),
                         ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'))
        X = X_first_term + X_second_term     
        reshaped_X = np.reshape(X, (1, d_tot**4))
        total_X = total_X + reshaped_X
    grad_QFI_parameters = total_X @ np.reshape(np.einsum('i...,j->ij...', jac_unitary_parameters_complex, variational_unitary_complex.conj()) 
                                     + np.einsum('i,j...->ij...', variational_unitary_complex, jac_unitary_parameters_complex.conj()), (d_tot**4, np.prod(shape_parameters)))
    grad_QFI_parameters= np.real(np.reshape(grad_QFI_parameters, shape_parameters))

    return np.real(grad_QFI_parameters)

# return sequential QFI, input state rho and list of control operations, for identical unitary control operations
# rho: initial input state
# control_parameters: variational parameters for initial control operation, shape: (num_layers, n_qubits, 3)
# E0_theta: Choi operator of the channel to estimate
# dE0_theta: derivative of E_theta
# N_steps: number of channels to estimate
# d: system dimension 
# d_a: ancilla dimension
# n_qubits: number of qubits for the variational circuit
# num_layers: number of layers for the variational circuit
# iterations: maximal number of iterations
# eps_SLD: numerical tolerance in rounding the nonzero denominator for computing the SLD
# eps_QFI: numerical tolerance for increase of QFI
# decay_parameter0: initial perturbation noise strength (adding an additional depolarizing noise for better convergence)
def sequential_QFI_same_unitary_control(rho, control_parameters, E0_theta, dE0_theta, N_steps, d, d_a, n_qubits, num_layers, iterations, eps_SLD, eps_QFI, decay_parameter0):
    d_tot = d*d_a
    E0_theta = np.reshape(E0_theta, (d, d, d, d))
    dE0_theta = np.reshape(dE0_theta, (d, d, d, d))
    rho = np.reshape(rho, (d, d_a, d, d_a)).astype(np.complex128)
    
    def depolarizing_E_theta(decay_parameter):
        Choi_depolarizing = E_theta(K_thetas_dep(0, 0, decay_parameter, decay_parameter, decay_parameter))
        Choi_depolarizing = np.reshape(Choi_depolarizing, (d, d, d, d))
        composed_Choi = contract(Choi_depolarizing, ('i3', 'i2', 'j3', 'j2'), E0_theta, ('i2', 'i1', 'j2', 'j1'), ('i3', 'i1', 'j3', 'j1'))
        return composed_Choi
    def ddepolarizing_E_theta(decay_parameter):
        Choi_depolarizing = E_theta(K_thetas_dep(0, 0, decay_parameter, decay_parameter, decay_parameter))
        Choi_depolarizing = np.reshape(Choi_depolarizing, (d, d, d, d))
        dcomposed_Choi = contract(Choi_depolarizing, ('i3', 'i2', 'j3', 'j2'), dE0_theta, ('i2', 'i1', 'j2', 'j1'), ('i3', 'i1', 'j3', 'j1'))
        return dcomposed_Choi
    
    # variational ansatz for single unitary control
    def variational_layer(parameters):
        for i in range(n_qubits):
            qml.Rot(parameters[i][0], parameters[i][1], parameters[i][2], wires=[i])
        if n_qubits >= 2:
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])
    
    def var_circuit(parameters):
        for i in range(num_layers):
            variational_layer(parameters[i])
            
    matrix_fn = qml.matrix(var_circuit, wire_order=[i for i in range(n_qubits)])
    
    def variational_unitary(control_parameters):
        control_unitary = matrix_fn(control_parameters)
        control_unitary_re = qml.numpy.reshape(qml.numpy.real(control_unitary), (1, d_tot, d_tot))
        control_unitary_im = qml.numpy.reshape(qml.numpy.imag(control_unitary), (1, d_tot, d_tot))
        variational_control_unitary = qml.numpy.concatenate([control_unitary_re, control_unitary_im])
        return variational_control_unitary          

    # The current value of QFI for comparison
    current_QFI = -1
    for item in range(iterations):
        
        # add the perturbation noise exponentially decaying with iterations (set decay rate to 0.8)
        decay_parameter = decay_parameter0*0.8**item
        E1_theta = depolarizing_E_theta(decay_parameter)
        dE1_theta = ddepolarizing_E_theta(decay_parameter)

        # circuit construction
        circuit = []
        for i in range(N_steps-1):
            circuit += [E1_theta, ('i{}'.format(2*N_steps-2*i), 'i{}'.format(2*N_steps-2*i-1), 'j{}'.format(2*N_steps-2*i), 'j{}'.format(2*N_steps-2*i-1)),
                       (d, d_a, d, d_a, d, d_a, d, d_a), ('i{}'.format(2*N_steps-2*i-1), 'ia{}'.format(N_steps-i), 'i{}'.format(2*N_steps-2*i-2), 'ia{}'.format(N_steps-i-1), 'j{}'.format(2*N_steps-2*i-1), 'ja{}'.format(N_steps-i), 'j{}'.format(2*N_steps-2*i-2), 'ja{}'.format(N_steps-i-1))]
        circuit += [E1_theta, ('i2', 'i1', 'j2', 'j1'), ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1')]
        circuit_expr = contract_expression(*circuit, constants=[2*i for i in range(N_steps)])
    
        # derivative of circuit
        dE_tensor_network = [np.reshape(np.vstack((E1_theta, dE1_theta)), (2, d, d, d, d)), ('k1', 'i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1))] 
        for i in range(N_steps-2):
            dE_tensor_network += [np.reshape(np.vstack((E1_theta, dE1_theta, np.zeros((d, d, d, d)), E1_theta)), (2, 2, d, d, d, d)), ('k{}'.format(i+1), 'k{}'.format(i+2), 'i{}'.format(2*N_steps-2*i-2), 'i{}'.format(2*N_steps-2*i-3), 'j{}'.format(2*N_steps-2*i-2), 'j{}'.format(2*N_steps-2*i-3))]
        dE_tensor_network += [np.reshape(np.vstack((dE1_theta, E1_theta)), (2, d, d, d, d)), ('k{}'.format(N_steps-1), 'i2', 'i1', 'j2', 'j1')]
        dcircuit = []
        for i in range(N_steps-1):
            dcircuit += [dE_tensor_network[2*i], dE_tensor_network[2*i+1], (d, d_a, d, d_a, d, d_a, d, d_a), ('i{}'.format(2*N_steps-2*i-1), 'ia{}'.format(N_steps-i), 'i{}'.format(2*N_steps-2*i-2), 'ia{}'.format(N_steps-i-1), 'j{}'.format(2*N_steps-2*i-1), 'ja{}'.format(N_steps-i), 'j{}'.format(2*N_steps-2*i-2), 'ja{}'.format(N_steps-i-1))]
        dcircuit += [dE_tensor_network[2*N_steps-2], dE_tensor_network[2*N_steps-1], ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1')]
        dcircuit_expr = contract_expression(*dcircuit, constants=[2*i for i in range(N_steps)])
        
        control_unitary = variational_unitary(control_parameters)
        control_unitary_complex = control_unitary[0] + 1j*control_unitary[1]
        control_Choi_vec = qml.numpy.reshape(control_unitary_complex, (d_tot**2, 1))
        control_Choi = control_Choi_vec @ control_Choi_vec.conj().T
        reshaped_control = np.reshape(control_Choi, (d, d_a, d, d_a, d, d_a, d, d_a))
        controls = [reshaped_control]*(N_steps-1)

        circuit_tensors = reversed(controls)
        circuit_tensor = circuit_expr(*circuit_tensors)
        
        # output state
        rho_out = contract(circuit_tensor, ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1'),
                           rho, ('i1', 'ia1', 'j1', 'ja1'), ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps)))

        dcircuit_tensors = reversed(controls)
        dcircuit_tensor = dcircuit_expr(*dcircuit_tensors)

        # derivative of output state
        drho_out = contract(dcircuit_tensor, ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1'),
                           rho, ('i1', 'ia1', 'j1', 'ja1'), ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps)))    
        
        # SLD
        rho_out = np.reshape(rho_out, (d_tot, d_tot))
        drho_out = np.reshape(drho_out, (d_tot, d_tot))
        eigvalues, eigvectors = np.linalg.eigh(rho_out)
        reshaped_eigvectors = []
        for i in range(d_tot):
            reshaped_eigvectors.append(np.reshape(eigvectors[:, i], (d_tot, 1)))
        SLD_operator = np.zeros((d_tot, d_tot), dtype=complex)
        for k in range(d_tot):
            for l in range(d_tot):
                if abs(eigvalues[k] + eigvalues[l]) > eps_SLD:
                    SLD_element = ((2 / (eigvalues[k] + eigvalues[l])) * 
                                    (reshaped_eigvectors[k].conj().T @ drho_out @ reshaped_eigvectors[l]) 
                                    * (reshaped_eigvectors[k] @ reshaped_eigvectors[l].conj().T)) 
                else:
                    SLD_element = 0
                SLD_operator += SLD_element
        SLD_operator = np.reshape(SLD_operator, (d_tot, d_tot))
        
        # update current QFI
        QFI = np.real(np.trace(rho_out @ SLD_operator @ SLD_operator))
        if abs(QFI - current_QFI) < eps_QFI:
            return QFI, rho, control_parameters
        else:
            current_QFI = QFI
        if item%10 == 0:
            print('iteration: QFI', item, QFI)
                
        rho_out = np.reshape(rho_out, (d, d_a, d, d_a))
        drho_out = np.reshape(drho_out, (d, d_a, d, d_a))
                
        # find a new input state
        X_first_term = contract(np.reshape(2*SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)),
                                dcircuit_tensor, ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1'), ('j1', 'ja1', 'i1', 'ia1'))
        X_second_term = contract(np.reshape(-SLD_operator @ SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)),
                                circuit_tensor, ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'i1', 'ia1', 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'j1', 'ja1'), ('j1', 'ja1', 'i1', 'ia1'))
        X = X_first_term + X_second_term
        X = np.reshape(X, (d_tot, d_tot))
        eigvalues_X, eigvectors_X = np.linalg.eigh(X)
        psi_new = np.reshape(eigvectors_X[:, -1], (d_tot, 1))
        rho_new = psi_new @ psi_new.conj().T

        # update input state
        rho = rho_new
        rho = np.reshape(rho, (d, d_a, d, d_a))

        # gradient of cost function
        def grad_cost_fn(variational_parameters):
            return -grad_QFI_fn(variational_parameters, variational_unitary=variational_unitary, E_theta=E1_theta, dE_tensor_network=dE_tensor_network, rho=rho, SLD_operator=SLD_operator, d=d, d_a=d_a, N_steps=N_steps)

        # cost function
        def opt_cost(variational_parameters):
            control_unitary = variational_unitary(variational_parameters)
            control_unitary_complex = control_unitary[0] + 1j*control_unitary[1]
            control_Choi_vec = qml.numpy.reshape(control_unitary_complex, (d_tot**2, 1))
            control_Choi = control_Choi_vec @ control_Choi_vec.conj().T
            reshaped_control = np.reshape(control_Choi, (d, d_a, d, d_a, d, d_a, d, d_a))
            
            composed_transition_matrix = contract(E1_theta, ('i2', 'i1', 'j2', 'j1'),
                           reshaped_control, ('i1', 'ia1', 'i0', 'ia0', 'j1', 'ja1', 'j0', 'ja0'), 
                                                  ('i2', 'ia1', 'j2', 'ja1', 'i0', 'ia0', 'j0', 'ja0'))
            composed_transition_matrix = np.reshape(composed_transition_matrix, (d_tot**2, d_tot**2))
            composed_transition_matrix_power = np.linalg.matrix_power(composed_transition_matrix, N_steps-1)
            composed_transition_matrix_power = np.reshape(composed_transition_matrix_power, (d, d_a, d, d_a, d, d_a, d, d_a))
            X_second_term = contract(np.reshape(-SLD_operator @ SLD_operator, (d, d_a, d, d_a)), 
                                  ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)),
                composed_transition_matrix_power, ('i{}'.format(2*N_steps), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i2', 'ia1', 'j2', 'ja1'),
                      E1_theta, ('i2', 'i1', 'j2', 'j1'),  rho, ('i1', 'ia1', 'j1', 'ja1')) 
            dcomposed_transition_matrix = contract(np.reshape(np.vstack((E1_theta, dE1_theta, np.zeros((d, d, d, d)), E1_theta)), (2, 2, d, d, d, d)), ('k1', 'k2', 'i2', 'i1', 'j2', 'j1'), 
                         reshaped_control, ('i1', 'ia1', 'i0', 'ia0', 'j1', 'ja1', 'j0', 'ja0'),
                        ('k1', 'i2', 'ia1', 'j2', 'ja1', 'k2', 'i0', 'ia0', 'j0', 'ja0'))
            dcomposed_transition_matrix = np.reshape(dcomposed_transition_matrix, (2*d_tot**2, 2*d_tot**2))
            dcomposed_transition_matrix_power = np.linalg.matrix_power(dcomposed_transition_matrix, N_steps-2)
            dcomposed_transition_matrix_power = np.reshape(dcomposed_transition_matrix_power, (2, d, d_a, d, d_a, 2, d, d_a, d, d_a))
            X_first_term = contract(np.reshape(2*SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)),
                np.reshape(np.vstack((E1_theta, dE1_theta)), (2, d, d, d, d)), 
                        ('k1', 'i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1)),
                        reshaped_control, ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1)),
                        dcomposed_transition_matrix_power, ('k1', 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1), 'k{}'.format(N_steps-1), 'i2', 'ia1', 'j2', 'ja1'),
                       np.reshape(np.vstack((dE1_theta, E1_theta)), (2, d, d, d, d)), 
                        ('k{}'.format(N_steps-1), 'i2', 'i1', 'j2', 'j1'),
                       rho, ('i1', 'ia1', 'j1', 'ja1'))
            return -np.real(X_first_term + X_second_term)
            
        opt = qml.AdagradOptimizer(stepsize=0.2/N_steps)

        # gradient descent
        for i in range(200): # 200 iterations
            control_parameters, cost_ = opt.step_and_cost(opt_cost, control_parameters, grad_fn=grad_cost_fn)                
    
    return QFI, rho, control_parameters