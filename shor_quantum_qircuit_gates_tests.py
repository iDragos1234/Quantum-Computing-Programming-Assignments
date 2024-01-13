import numpy as np
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.circuit import Qubit

from my_quantum_circuit import MyQuantumCircuit


def _test_c_mult_mod_n_helper(a, b, n, n_qubits):
    control_qreg = QuantumRegister(1)
    qreg = QuantumRegister(n_qubits)

    aux_qreg = QuantumRegister(n_qubits)
    overflow_qreg = QuantumRegister(1)
    ancilla_qreg = QuantumRegister(1)

    creg = ClassicalRegister(n_qubits)

    circuit = QuantumCircuit(control_qreg, qreg, aux_qreg, overflow_qreg, ancilla_qreg, creg)

    aux_qreg_overflow_extension = aux_qreg[:] + overflow_qreg[:]

    my_circuit = ShorQuantumCircuit(circuit)
    my_circuit\
        .initialize_qreg(control_qreg, 1)\
        .initialize_qreg(qreg, a)\
        .c_mult_mod_n(control_qreg[0], qreg, aux_qreg_overflow_extension, b, n, ancilla_qreg[0])\
        .measure_qreg_to_creg(1, 0)\
        .execute_circuit(1)

    result = int(list(my_circuit.result.get_counts(circuit).keys())[0], 2)

    return result

def _test_c_mult_mod_n(n_qubits):
    flag = True
    for n in range(1, 2 ** n_qubits):
        for a in range(n):
            for b in range(n):
                result = _test_c_mult_mod_n(a, b, n, n_qubits=n_qubits)
                _ = (result == (a * b) % n)
                flag = flag and _
                if not _:
                    print(f'({a} + {b}) mod {n} = {result}')

    if not flag:
        raise Exception('Test failed')

def _test_phi_add_mod_n(a, b, n, n_qubits):
    control_qreg = QuantumRegister(2)
    qreg = QuantumRegister(n_qubits)
    overflow_qreg = QuantumRegister(1)
    ancilla_qreg = QuantumRegister(1)

    creg = ClassicalRegister(n_qubits)

    circuit = QuantumCircuit(control_qreg, qreg, overflow_qreg, ancilla_qreg, creg)

    qreg_overflow_extension = qreg[:] + overflow_qreg[:]

    my_circuit = ShorQuantumCircuit(circuit)

    my_circuit\
        .initialize_qreg(control_qreg, 3)\
        .initialize_qreg(qreg, b)\
        .quantum_fourier_transform(qreg_overflow_extension)\
        .cc_phi_add_mod_n(control_qreg[0], control_qreg[1], qreg_overflow_extension, a, n, ancilla_qreg[0])\
        .inverse_quantum_fourier_transform(qreg_overflow_extension)\
        .measure_qreg_to_creg(1, 0)\
        .execute_circuit(1)

    result = int(list(my_circuit.result.get_counts(circuit).keys())[0], 2)

    return result

def _test_phi_add_mod_n(n_qubits):
    flag = True
    for n in range(1, 2 ** n_qubits):
        for a in range(n):
            for b in range(n):
                result = _test_phi_add_mod_n(a, b, n, n_qubits=n_qubits)
                _ = (result == (a + b) % n)
                flag = flag and _
                if not _:
                    print(f'({a} + {b}) mod {n} = {result}')

    if not flag:
        raise Exception('Test failed')

def _test_phi_add(n_qubits):
    flag = True
    n_states = 2 ** n_qubits
    for a in range(n_states):
        for b in range(n_states):

            qreg = QuantumRegister(n_qubits)
            creg = ClassicalRegister(n_qubits)

            circuit = QuantumCircuit(qreg, creg)

            my_circuit = ShorQuantumCircuit(circuit)
            my_circuit\
                .initialize_qreg(qreg, b)\
                .quantum_fourier_transform(qreg)\
                .phi_add(a, qreg)\
                .inverse_quantum_fourier_transform(qreg)\
                .measure_qreg_to_creg(0, 0)\
                .execute_circuit(1)

            result = int(list(my_circuit.result.get_counts(circuit).keys())[0], 2)
            flag = flag and (result == (a + b) % n_states)
    if not flag:
        raise Exception('Phi Addition failed.')


def tests_run():
    _test_phi_add_mod_n(n_qubits=3)
    _test_c_mult_mod_n(n_qubits=3)
    _test_phi_add(n_qubits=4)
