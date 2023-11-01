import numpy as np
from numpy import pi as pi

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, execute
from qiskit.result import Result
from qiskit.tools.monitor import job_monitor

from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from utils import _statevector_to_circle_notation


class MyQuantumCircuit:
    circuit: QuantumCircuit = None
    job = None
    result: Result = None

    def __init__(self, circuit=None):
        if circuit is not None:
            self.circuit = circuit
        pass

    def create_circuit(
            self,
            qregs: list[QuantumRegister],
            cregs: list[ClassicalRegister],
            name=None
    ):
        self.circuit = QuantumCircuit(*(qregs + cregs), name=name)
        return self

    def create_simple_circuit(self, n_qubits, name=None):
        qreg = QuantumRegister(n_qubits)
        creg = ClassicalRegister(n_qubits)
        return self.create_circuit([qreg], [creg], name)

    def prepare_qreg_step(
            self,
            qreg_index: int,
            init_val: int = None,
            had_gate_indices: list[int] = None
    ):
        qreg = self.circuit.qregs[qreg_index]
        qreg_size = int(qreg.size)
        n_states = 2 ** qreg_size

        # Pre-conditions:
        if init_val is not None:
            assert 0 <= init_val < n_states
        if had_gate_indices is not None:
            assert len(had_gate_indices) <= qreg_size

        # Initialization step - optionally initialize with given value:
        if init_val is not None:
            init_val_bitstring = format(init_val, f'0{qreg_size}b')
            init_val_statevector = Statevector.from_label(init_val_bitstring)
            self.circuit.initialize(init_val_statevector, qreg)

        # Initialization step - optionally apply Hadamard gates on selected qubits:
        if had_gate_indices is not None:
            had_gate_indices = list(set(had_gate_indices))  # Ensure that HAD is not applied at most once for each qubit.
            self.circuit.h(qreg[had_gate_indices])

        return self

    def save_statevector(self, statevector_label=None):
        self.circuit.save_statevector(label=statevector_label)
        return self

    def measure_qreg_to_creg(self, qreg_index: int, creg_index: int):
        self.circuit.measure(self.circuit.qregs[qreg_index], self.circuit.cregs[creg_index])
        return self

    def execute_circuit(self, N_SHOTS=None):
        """
        Execute this circuit on the QASM simulator.
        :param N_SHOTS: number of shots
        :return: this circuit
        """
        simulator = Aer.get_backend('qasm_simulator')
        self.job = execute(self.circuit, backend=simulator, shots=N_SHOTS)
        self.result = self.job.result()
        return self

    def exec_circ_on_ibm_qc(
            self,
            provider,
            IBM_QUANTUM_COMPUTER: str = 'ibm_nairobi',
            N_SHOTS = None
    ):
        """
        Execute this circuit on actual IBM quantum computer.
        Note that a job on an IBM cloud QC can take a while to execute (usually hours).
        :param provider: (simply do provider = IBMQ.load_account())
        :param IBM_QUANTUM_COMPUTER: the id (name, label etc.) of the IBM quantum computer to execute this circuit on
        :param N_SHOTS: number of shots
        :return: this circuit
        """
        qcomp = provider.get_backend(IBM_QUANTUM_COMPUTER)
        self.job = execute(self.circuit, backend=qcomp, shots=N_SHOTS)
        self.result = self.job.result()
        return self

    def get_job_monitor(self):
        return self, job_monitor(self.job)

    def draw_circuit(self):
        print(self.circuit)
        return self

    def draw_circuit_nice(self):
        return self, self.circuit.draw('mpl')

    def plot_statevector(self, statevector_label):
        print(f'Statevector \"{statevector_label}\": ')
        _statevector_to_circle_notation(self.result.data()[statevector_label])
        return self

    def plot_results(self):
        return self, plot_histogram(self.result.get_counts(self.circuit))

    def barrier(self):
        self.circuit.barrier()
        return self
