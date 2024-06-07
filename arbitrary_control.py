# tensor network optimization for arbitrary control operations

from functions import *

# return sequential QFI, input state rho and list of control operations, for arbitrary control operations
# rho: initial input state
# controls: list of Choi operators of initial control operations
# E0_theta: Choi operator of the channel to estimate
# dE0_theta: derivative of E_theta
# N_steps: number of channels to estimate
# d: system dimension 
# d_a: ancilla dimension
# iterations: maximal number of iterations
# eps_SLD: numerical tolerance in rounding the nonzero denominator for computing the SLD
# eps_QFI: numerical tolerance for increase of QFI
# decay_parameter0: initial perturbation noise strength (adding an additional depolarizing noise for better convergence)
def sequential_QFI_arbitrary_control(rho, controls, E0_theta, dE0_theta, N_steps, d, d_a, iterations, eps_SLD, eps_QFI, decay_parameter0):
    d_tot = d*d_a
    rho = np.reshape(rho, (d, d_a, d, d_a))
    E0_theta = np.reshape(E0_theta, (d, d, d, d))
    dE0_theta = np.reshape(dE0_theta, (d, d, d, d))

    # add perturbation noise to E_theta
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

    # SDP construction for control optimization
    Parameter_X = cp.Parameter((d_tot**2, d_tot**2), hermitian=True)
    control_to_optimize = cp.Variable((d_tot**2, d_tot**2), hermitian=True)
    obj_control = cp.Maximize(cp.real(cp.trace(control_to_optimize @ Parameter_X)))
    constraints_control = [control_to_optimize >> 0, cp.partial_trace(control_to_optimize, [d_tot, d_tot], axis=0) - np.eye(d_tot) == 0]
    opt_control_prob = cp.Problem(obj_control, constraints_control)

    # current value of QFI
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
            return QFI, rho, controls
        else:
            current_QFI = QFI
        if item%10 == 0:
            print('iteration', item, 'QFI', QFI)
        
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

        # save intermediates for updating the control
        end_tensor_2 = contract(np.reshape(-SLD_operator @ SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)), 
                            E1_theta, ('i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1)), 
                            ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)))
        end_tensor_1 = contract(np.reshape(2*SLD_operator, (d, d_a, d, d_a)), ('j{}'.format(2*N_steps), 'ja{}'.format(N_steps), 'i{}'.format(2*N_steps), 'ia{}'.format(N_steps)), 
                            dE_tensor_network[0], ('k1', 'i{}'.format(2*N_steps), 'i{}'.format(2*N_steps-1), 'j{}'.format(2*N_steps), 'j{}'.format(2*N_steps-1)),
                            ('k1', 'i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)))
        
        later_circuit_tensors_2 = [end_tensor_2]
        later_circuit_tensors_1 = [end_tensor_1]
        for i in range(N_steps-2):
            end_tensor_2 = contract(end_tensor_2, ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)), 
                        controls[-i-1], ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1)),
                                     E1_theta, ('i{}'.format(2*N_steps-2), 'i{}'.format(2*N_steps-3), 'j{}'.format(2*N_steps-2), 'j{}'.format(2*N_steps-3)),
                                    ('i{}'.format(2*N_steps-3), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-3), 'ja{}'.format(N_steps-1)))
            later_circuit_tensors_2.append(end_tensor_2)
            end_tensor_1 = contract(end_tensor_1, ('k1', 'i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps)), 
                        controls[-i-1], ('i{}'.format(2*N_steps-1), 'ia{}'.format(N_steps), 'i{}'.format(2*N_steps-2), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-1), 'ja{}'.format(N_steps), 'j{}'.format(2*N_steps-2), 'ja{}'.format(N_steps-1)),
                                     dE_tensor_network[2], ('k1', 'k2', 'i{}'.format(2*N_steps-2), 'i{}'.format(2*N_steps-3), 'j{}'.format(2*N_steps-2), 'j{}'.format(2*N_steps-3)),
                                    ('k2', 'i{}'.format(2*N_steps-3), 'ia{}'.format(N_steps-1), 'j{}'.format(2*N_steps-3), 'ja{}'.format(N_steps-1)))
            later_circuit_tensors_1.append(end_tensor_1)

        start_tensor_2 = contract(E1_theta, ('i2', 'i1', 'j2', 'j1'), rho, ('i1', 'ia1', 'j1', 'ja1'),
                              ('i2', 'ia1', 'j2', 'ja1'))

        start_tensor_1 = contract(dE_tensor_network[-2], ('k', 'i2', 'i1', 'j2', 'j1'), rho, ('i1', 'ia1', 'j1', 'ja1'),
                              ('k', 'i2', 'ia1', 'j2', 'ja1'))

        # update the control
        for control_index in range(N_steps-1):
            end_tensor_2 = later_circuit_tensors_2[-control_index-1]
            end_tensor_1 = later_circuit_tensors_1[-control_index-1]
            if control_index != 0:
                start_tensor_2 = contract(E1_theta, ('i4', 'i3', 'j4', 'j3'),
                    controls[control_index-1], ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'),
                    start_tensor_2, ('i2', 'ia1', 'j2', 'ja1'), ('i4', 'ia2', 'j4', 'ja2'))
                start_tensor_1 = contract(dE_tensor_network[2], ('k1', 'k2', 'i4', 'i3', 'j4', 'j3'),
                    controls[control_index-1], ('i3', 'ia2', 'i2', 'ia1', 'j3', 'ja2', 'j2', 'ja1'),
                    start_tensor_1, ('k2', 'i2', 'ia1', 'j2', 'ja1'), ('k1', 'i4', 'ia2', 'j4', 'ja2'))
            X_second_term = contract(end_tensor_2, ('i3', 'ia2', 'j3', 'ja2'),
                             start_tensor_2, ('i2', 'ia1', 'j2', 'ja1'),
                             ('j3', 'ja2', 'j2', 'ja1', 'i3', 'ia2', 'i2', 'ia1'))
            X_first_term = contract(end_tensor_1, ('k', 'i3', 'ia2', 'j3', 'ja2'),
                             start_tensor_1, ('k', 'i2', 'ia1', 'j2', 'ja1'),
                             ('j3', 'ja2', 'j2', 'ja1', 'i3', 'ia2', 'i2', 'ia1')) 
            X = X_first_term + X_second_term
            reshaped_X = np.reshape(X, (d_tot**2, d_tot**2))
            hermitian_X = 0.5*(reshaped_X + reshaped_X.conj().T)
            Parameter_X.value = hermitian_X
            opt_control_prob.solve(solver=cp.MOSEK, warm_start=True)
            control_new = control_to_optimize.value
            controls[control_index] = np.reshape(control_new, (d, d_a, d, d_a, d, d_a, d, d_a))

    # return the optimized QFI, input state and list of control operations in the Choi operator forms
    return QFI, rho, controls
