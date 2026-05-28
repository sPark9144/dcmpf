import numpy as np
import pickle, os

import threading, time

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate, UGate, XGate, HGate, CHGate, CSwapGate, RXXGate, RYYGate, RZZGate

from qiskit.quantum_info import Statevector, Operator, partial_trace, DensityMatrix

from qiskit import transpile

from qiskit.exceptions import QiskitError

from qiskit.visualization import plot_histogram

#####################################################

# Load data
def load(file_name:str):
    with open(file=file_name, mode='rb') as f:
        info, data = pickle.load(f)
    return info, data

def load_pickle(file_name:str):
    with open(file=file_name+'.pkl', mode='rb') as f:
        info, data = pickle.load(f)
    return info, data

# Save data as pickle
def save_pickle(file_name:str, info, data):
    with open(file=file_name+'.pkl', mode='wb') as f:
        pickle.dump((info, data), f)

# Relative error

def rel_error(a, b) : 
    return np.abs((a-b)/a)

def abs_error(a, b) : 
    return np.abs(a-b)

##################################################### Exact time evolution 

# Hamiltonian 

def mat_for_ham(nq, axis: str, j) : 
    if axis == 'xx' :
        qc=QuantumCircuit(nq)
        qc.x(j)
        qc.x(j+1)
        mat=Operator(qc).data
    elif axis == 'yy' :
        qc=QuantumCircuit(nq)
        qc.y(j)
        qc.y(j+1)
        mat=Operator(qc).data
    elif axis == 'zz' :
        qc=QuantumCircuit(nq)
        qc.z(j)
        qc.z(j+1)
        mat=Operator(qc).data
    elif axis == 'x' :
        qc=QuantumCircuit(nq)
        qc.x(j)
        mat=Operator(qc).data
    elif axis == 'y' :
        qc=QuantumCircuit(nq)
        qc.y(j)
        mat=Operator(qc).data
    elif axis == 'z' :
        qc=QuantumCircuit(nq)
        qc.z(j)
        mat=Operator(qc).data   
    else: 
        print('Invalid axis')
        mat = None
    return mat

def hamiltonian_Heisenberg_mat(nq, coeffs: dict) : 
    ham=0
    for axis in list(coeffs.keys()) : 
        for j in range(nq+1-len(axis)):
            ham = ham + (coeffs[axis])*mat_for_ham(nq, axis, j)
    return ham

# Time evolution matrix 

def time_evolution_mat(ham_mat, t): 
    vals, uni = np.linalg.eig(ham_mat)
    exp_vals = np.exp(-1j*t*vals)
    exp_diag = np.diag(exp_vals)

    te_mat = uni@exp_diag@np.conjugate(np.transpose(uni))

    return te_mat

# State vector

def state_to_vec(nq, circ) : 
    qc=QuantumCircuit(nq)
    add_circuit(qc, circ)

    stat = Statevector(qc)
    vec = stat.data

    return vec

# Observable 

def mat_for_obs(nq, axis: str, pos: list) : 
    if axis == 'xx' :
        qc=QuantumCircuit(nq)
        qc.x(pos[0])
        qc.x(pos[1])
        mat=Operator(qc).data
    elif axis == 'yy' :
        qc=QuantumCircuit(nq)
        qc.y(pos[0])
        qc.y(pos[1])
        mat=Operator(qc).data
    elif axis == 'zz' :
        qc=QuantumCircuit(nq)
        qc.z(pos[0])
        qc.z(pos[1])
        mat=Operator(qc).data
    elif axis == 'x' :
        qc=QuantumCircuit(nq)
        qc.x(pos[0])
        mat=Operator(qc).data
    elif axis == 'y' :
        qc=QuantumCircuit(nq)
        qc.y(pos[0])
        mat=Operator(qc).data
    elif axis == 'z' :
        qc=QuantumCircuit(nq)
        qc.z(pos[0])
        mat=Operator(qc).data   
    else: 
        print('Invalid axis')
        mat = None
    return mat

def observable_mat(nq, obs_dict: dict) : 
    axis = (list(obs_dict.keys()))[0]
    n_obs = len(list(obs_dict.values())[0])
    if len(axis) == 2 : 
        mat = 0
        for pos in obs_dict[axis] : 
            mat = mat + mat_for_obs(nq, axis, pos)
    elif len(axis) == 1 :
        mat = 0
        for pos in obs_dict[axis] : 
            mat = mat + mat_for_obs(nq, axis, [pos])
    return mat/n_obs


#####################################################  Trotter circuits

# Add gates to the circuit

def add_circuit(qc, c_list, inv = False) : 
    if inv :
        for gate, qubit in reversed(c_list) : 
            qc.append(gate.inverse(), qubit)
    else : 
        for gate, qubit in c_list : 
            qc.append(gate, qubit)
    
# Trotter circuit for Heisenberg spin chain. Return the list with (gate, qubit) elements.
def block_two_q(dt, k, j, axis: str) : 
    if axis == 'xx' :
        circuits = [(HGate(), [k]), (HGate(), [k+j]), 
            (CXGate(), [k,k+j]), (RZGate(dt), [k+j]), (CXGate(), [k,k+j]), 
            (HGate(), [k]), (HGate(), [k+j])]
    elif axis == 'yy' : 
        circuits = [(RXGate(np.pi/2), [k]), (RXGate(np.pi/2), [k+j]), 
            (CXGate(), [k,k+j]), (RZGate(dt), [k+j]), (CXGate(), [k,k+j]), 
            (RXGate(-np.pi/2), [k]), (RXGate(-np.pi/2), [k+j])]
    elif axis == 'zz' : 
        circuits = [(CXGate(), [k,k+j]), (RZGate(dt), [k+j]), (CXGate(), [k,k+j])]
    else : 
        print('Invalid axis')
        circuits = None
    return circuits

def block_two_q_RSS(dt, k, j, axis: str) : 
    if axis == 'xx' :
        circuits = [(RXXGate(dt), [k, k+j])]
    elif axis == 'yy' : 
        circuits = [(RYYGate(dt), [k, k+j])]
    elif axis == 'zz' : 
        circuits = [(RZZGate(dt), [k, k+j])]
    else : 
        print('Invalid axis')
        circuits = None
    return circuits

def trotter_block_two_q(strength, dt, nq, k, axis: str) : 
    l_t_xx=[]
    for j in range(nq-1) : 
        l_t_xx.extend(block_two_q(-2*strength*dt, k+j, 1, axis))
    return l_t_xx

def trotter_block_one_q(strength, dt, nq, k, axis: str) : 
    l_t_x = []
    if axis == 'x' :
        for j in range(nq) : 
            l_t_x.append((RXGate(-2*strength*dt), [k+j]))
    elif axis == 'y' :
        for j in range(nq) : 
            l_t_x.append((RYGate(-2*strength*dt), [k+j]))
    elif axis == 'z' :
        for j in range(nq) : 
            l_t_x.append((RZGate(-2*strength*dt), [k+j]))
    else : 
        print('Invalid axis')
    return l_t_x

# For the XXZ Heisenberg model, with external field. The coefficients are given by dictionary, as {'xx' : 0.5, 'yy' : 0.3, ...}

def trotter_second_order_XXZ(coeffs: dict, nq, t, n, k) : 
    dt=t/n
    base = []
    for axis in ['xx', 'yy', 'zz', 'z', 'yy', 'xx'] : 
        if axis in ['xx' ,'yy']: 
            base += trotter_block_two_q(coeffs[axis], dt/2, nq, k, axis)
        elif axis =='zz' : 
            base += trotter_block_two_q(coeffs[axis], dt, nq, k, axis)
        else : 
            base += trotter_block_one_q(coeffs[axis], dt, nq, k, axis)
    circuits = base*n
    return circuits

def trotter_first_order_XXZ(coeffs: dict, ord: list, nq, t, n, k) : 
    dt = t/n
    base = []
    for axis in ord : 
        if axis in ['xx', 'yy', 'zz'] : 
            base += trotter_block_two_q(coeffs[axis], dt, nq, k, axis)
        else : 
            base += trotter_block_one_q(coeffs[axis], dt, nq, k, axis)
    circuits = base*n
    return circuits

# Initial state preparation 

def init_state_prep(nq, init: str ,k=0) :
    if init in ['00', 'state_vector'] : 
        circuits = []
    elif init == '11' : 
        circuits = [] 
        for i in range(nq) : 
            circuits.append((XGate(), [k+i]))
    elif init == '++' : 
        circuits = [] 
        for i in range(nq) : 
            circuits.append((HGate(), [k+i]))

    elif init == 'neel' : 
        circuits = []
        for i in range(0,nq,2) :
            circuits.append((XGate(), [k+i]))
    else : 
        print('Invalid initial state')
        circuits = None
    return circuits 

# Observables
def measurement_basis_rot(nq, measure_basis: str, k) : 
    if measure_basis in ['z', 'zz', 'state_vector'] :
        circuits = []
    elif measure_basis in ['x', 'xx'] : 
        circuits=[]
        for i in range(nq) : 
             circuits.append((HGate(), [k+i]))
    else : 
        print('Invalid basis')
        circuits = None
    return circuits

# Full circuit

# Full circuit with measurement along the desired basis

def trotter_second_XXZ(nq, init: str, coeffs: dict, t, n, measure_basis: str, initvec, k=0) : 
    if measure_basis == 'state_vector' :
        qc = QuantumCircuit(nq)
    else : 
        qc = QuantumCircuit(nq, nq)
    
    if init == 'state_vector' :
        qc.prepare_state(initvec)
    init_circ = init_state_prep(nq, init, k)
    trott_circ = trotter_second_order_XXZ(coeffs, nq, t, n ,k)
    measure_circ = measurement_basis_rot(nq, measure_basis, k)

    add_circuit(qc, init_circ)
    add_circuit(qc, trott_circ)
    add_circuit(qc, measure_circ)
    
    if measure_basis != 'state_vector' :
        qc.measure(range(nq),range(nq))
    
    qc.save_density_matrix()

    return qc

def trotter_first_XXZ(nq, init: str, coeffs: dict, ord: list, t, n, measure_basis: str, initvec, k=0) : 
    if measure_basis == 'state_vector' :
        qc = QuantumCircuit(nq)
    else : 
        qc = QuantumCircuit(nq, nq)
    if init == 'state_vector' :
        qc.prepare_state(initvec)
    init_circ = init_state_prep(nq, init, k)
    trott_circ = trotter_first_order_XXZ(coeffs, ord, nq, t, n ,k)
    measure_circ = measurement_basis_rot(nq, measure_basis, k)

    add_circuit(qc, init_circ)
    add_circuit(qc, trott_circ)
    add_circuit(qc, measure_circ)

    if measure_basis != 'state_vector' :
        qc.measure(range(nq),range(nq))

    qc.save_density_matrix()

    return qc

# Noiseless run

# def trotter_second_XXZ_shotnoiseless(nq, init: str, coeffs: dict, t, n) : 
#     qc = QuantumCircuit(nq,nq)

#     init_circ = init_state_prep(nq, init, 0)
#     trott_circ = trotter_second_order_XXZ(coeffs, nq, t, n ,0)

#     add_circuit(qc, init_circ)
#     add_circuit(qc, trott_circ)

#     stat = Statevector(qc)
#     vec = stat.data

#     return vec

# def trotter_first_XXZ_shotnoiseless(nq, init: str, coeffs: dict, ord: list, t, n) : 
#     qc = QuantumCircuit(nq,nq)

#     init_circ = init_state_prep(nq, init)
#     trott_circ = trotter_first_order_XXZ(coeffs, ord, nq, t, n, 0)

#     add_circuit(qc, init_circ)
#     add_circuit(qc, trott_circ)

#     stat = Statevector(qc)
#     vec = stat.data

#     return vec

# Circuit run
# service = QiskitRuntimeService()
# backend_yonsei = service.backend("ibm_yonsei")


# The run_.._simulator can produce all possible result: (initial state as vector, as circuit) x (output as measurement result, as statevector). 
# If initial state is given by vector, then initvec = the vector (and in the circuit, the initial state index has to be 'state_vector'), otherwise it is None.

def run_circuit_on_simulator(circuit, opt_level, simulator, n_shots, initvec, output_statevec) :
    circuit_transpiled = transpile(circuit, simulator, optimization_level=opt_level)

    if initvec is None:
        circuit_run = simulator.run(circuit_transpiled, shots = n_shots )
    else:
        dm = np.outer(initvec, np.conjugate(initvec))
        dm_in = DensityMatrix(dm)
        circuit_run = simulator.run(circuit_transpiled, initial_state = dm_in, shots = n_shots)

    res = circuit_run.result()

    if not output_statevec : 
        return res.get_counts()
    
    data0 = res.data(0)  

    for key in ("density_matrix", "final_density_matrix", "statevector", "final_statevector"):
        if key in data0:
            arr = np.array(data0[key])

            if "statevector" in key:
                arr = arr.reshape(-1)  # just in case
                return np.outer(arr, np.conjugate(arr))
            else:
                return arr

    raise ValueError(
        "Simulator result has no density/statevector key. "
        f"Available keys in result.data(0): {list(data0.keys())}"
    )

    # if output_statevec :   
    #     try : 
    #         res_state = res.get_statevector()
    #         result = res_state.data
    #         return result
    #     except Exception :
    #         pass
    #     try : 
    #         res_dm = res.data(0).get('density_matrix', None)
    #         if res_dm is not None : 
    #             return np.array(res_dm)
    #     except Exception :
    #         pass
    #     raise ValueError("Simulator does not provide statevector or density_matrix.")
    
    # else :
    #     return res.get_counts()

# def run_circuit_on_simulator_initprep_sep(init_vec ,circuit, opt_level, simulator, n_shots) :
#     circuit_transpiled = transpile(circuit, simulator, optimization_level=opt_level)
#     circuit_run = simulator.run(circuit_transpiled, initial_state = init_vec, shots = n_shots)
#     circuit_result = circuit_run.result()

#     circuit_result_counts = circuit_result.get_counts()

#     return circuit_result_counts

def job_manager(job_id:str, service, time_gap=60) : 
    job =service.job(job_id)
    jobstatus = str(job.status())

    while not 'DONE' in jobstatus and not 'ERROR' in jobstatus : 
        time.sleep(time_gap)
        job = service.job(job_id)
        jobstatus = str(job.status())
    return jobstatus

def run_circuit_on_ibmq(circuit, opt_level, serv, bckend, n_shots) :
    circuit_transpiled = transpile(circuit, backend=bckend, optimization_level=opt_level)

    job = sampler.run(circuit_transpiled, shots=n_shots)
    job_id = job.job_id()

    jobstatus = job_manager(job_id, serv, time_gap=60)

    while 'ERROR' in jobstatus :
        job = sampler.run(circuit_transpiled, shots=n_shots)
        job_id = job.job_id()
        jobstatus = job_manager(job_id, serv, time_gap=60)

    results = job.result()
    counts_list = results.data.c.get_counts()

    return counts_list

# Post processing
# Expectation value, of mono Pauli elements (e.g. XX, ZZZZ, ..., not like XZX)
def expect_from_prob_mono_single(res: dict, obs_pos: list) : 
    n_shots=sum(res.values())
    rescopy=res.copy()
    bits_list = list(rescopy.keys())
    for bits in bits_list :
        for pos in obs_pos : 
            if bits[pos]=='1' : 
                rescopy[bits] = -rescopy[bits]
    expec=(sum(rescopy.values()))/n_shots
    return expec

def expect_from_prob_mono(res: dict, obs_dict: dict) : 
    axis = (list(obs_dict.keys()))[0]
    n_obs = len(list(obs_dict.values())[0])
    if len(axis) == 2 : 
        exp = 0
        for pos in obs_dict[axis] : 
            exp = exp + expect_from_prob_mono_single(res, pos)
    elif len(axis) == 1 :
        exp = 0
        for pos in obs_dict[axis] : 
            exp = exp + expect_from_prob_mono_single(res, [pos])
    return exp/n_obs


##################################################### MPF

def Vandermonde_even_mat(exponents: list) :
    dim = len(exponents)
    mat=[]
    for i in range(dim) : 
        row = []
        for j in range(dim) : 
            row.append((exponents[j])**(-2*(i)))
        mat.append(row)
    vandmat=np.array(mat)
    return vandmat

def MPF_even_coeff(exponents: list) : 
    dim = len(exponents)
    vandmat_inv = np.linalg.inv(Vandermonde_even_mat(exponents))
    sol_list = [1]+[0]*(dim-1)
    sol_vec=np.array(sol_list)
    coeff=vandmat_inv@sol_vec
    return coeff

# def MPF_conventional_shotnoiseless(nq, init: str, coeffs: dict, t, exponents: list, obs_dict: dict) : 
#     mpf_coeffs = MPF_even_coeff(exponents)
#     n_exponents = len(exponents)
#     obs = observable_mat(nq, obs_dict)

#     exp_list = []
#     for i in range(n_exponents) : 
#         state_vec = trotter_second_XXZ_shotnoiseless(nq, init, coeffs, t, exponents[i])
#         exp = np.real(np.conjugate(state_vec)@obs@state_vec)
#         exp_list.append(exp)

#     mpf_res = mpf_coeffs@(np.array(exp_list))

#     return mpf_res

# def MPF_dual_shotnoiseless(nq, init: str, coeffs: dict, t, exponents: list, obs_dict: dict) : 
#     mpf_coeffs = MPF_even_coeff(exponents)
#     n_exponents = len(exponents)
#     obs = observable_mat(nq, obs_dict)
#     order_original = list(coeffs.keys())
#     order_reversed = order_original[::-1]

#     exp_list = []
#     for i in range(n_exponents) : 
#         state_vec_1 = trotter_first_XXZ_shotnoiseless(nq, init, coeffs, order_original, t, exponents[i])
#         state_vec_2 = trotter_first_XXZ_shotnoiseless(nq, init, coeffs, order_reversed, t, exponents[i])
#         exp_1 = np.real(np.conjugate(state_vec_1)@obs@state_vec_1)
#         exp_2 = np.real(np.conjugate(state_vec_2)@obs@state_vec_2)
#         exp_list.append((exp_1+exp_2)/2)

#     mpf_res = mpf_coeffs@(np.array(exp_list))

#     return mpf_res


def MPF_conventional(nq, init: str, coeffs: dict, t, exponents: list, obs_dict: dict, simulator, opt_level, n_shots, initvec, output_statevec, k=0) : 
    mpf_coeffs = MPF_even_coeff(exponents)
    if output_statevec : 
        measure_axis = 'state_vector'
    else :
        measure_axis = (list(obs_dict.keys()))[0]
    obs = observable_mat(nq, obs_dict)
    n_exponents = len(exponents)

    exp_list = []
    for i in range(n_exponents) : 
        circ = trotter_second_XXZ(nq, init, coeffs, t, exponents[i], measure_axis, initvec, k)
        res = run_circuit_on_simulator(circ, opt_level, simulator, n_shots, initvec, output_statevec) 
        if not output_statevec : 
            exp = expect_from_prob_mono(res, obs_dict)
        else : 
            exp = np.real(np.trace(res@obs))
        exp_list.append(exp)

    mpf_res = mpf_coeffs@(np.array(exp_list))

    return mpf_coeffs, exp_list, mpf_res

def MPF_dual(nq, init: str, coeffs: dict, t, exponents: list, obs_dict: dict, simulator, opt_level, n_shots, initvec, output_statevec, k=0) : 
    mpf_coeffs = MPF_even_coeff(exponents)
    if output_statevec : 
        measure_axis = 'state_vector'
    else :
        measure_axis = (list(obs_dict.keys()))[0]
    obs = observable_mat(nq, obs_dict)
    n_exponents = len(exponents)

    order_original = list(coeffs.keys())
    order_reversed = order_original[::-1]

    exp_list = []
    for i in range(n_exponents) : 
        circ_1 = trotter_first_XXZ(nq, init, coeffs, order_original, t, exponents[i], measure_axis, initvec, k)
        circ_2 = trotter_first_XXZ(nq, init, coeffs, order_reversed, t, exponents[i], measure_axis, initvec, k)
        res_1 = run_circuit_on_simulator(circ_1, opt_level, simulator, int(n_shots/2), initvec, output_statevec)
        res_2 = run_circuit_on_simulator(circ_2, opt_level, simulator, int(n_shots/2), initvec, output_statevec) 
        if not output_statevec : 
            exp_1 = expect_from_prob_mono(res_1, obs_dict)
            exp_2 = expect_from_prob_mono(res_2, obs_dict)
        else : 
            exp_1 = np.real(np.trace(res_1@obs))
            exp_2 = np.real(np.trace(res_2@obs))
        
        exp_list.append((exp_1+exp_2)/2)

    mpf_res = mpf_coeffs@(np.array(exp_list))

    return mpf_coeffs, exp_list, mpf_res
