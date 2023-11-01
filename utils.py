import numpy as np
from numpy import pi as pi

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, execute
from qiskit.result import Result

from matplotlib.patches import Circle
import matplotlib.pyplot as plt


def _statevector_to_circle_notation(statevector: Statevector):
    RADIUS = 0.5
    n_states = statevector.data.size

    rows = int(np.ceil(n_states / 8.0))
    cols = min(n_states, 8)

    for row in range(rows):
        fig, axs = plt.subplots(1, cols)
        for col in range(cols):
            state_index = 8 * row + col
            state = statevector[state_index]
            probability = np.absolute(state)
            phase = np.angle(state)
            # amplitude area
            circleExt = Circle((RADIUS, RADIUS), RADIUS, color='gray', alpha=0.1)
            circleInt = Circle((RADIUS, RADIUS), probability / 2, color='b', alpha=0.3)
            axs[col].add_patch(circleExt)
            axs[col].add_patch(circleInt)
            axs[col].set_aspect('equal')
            state_number = "|" + str(state_index) + ">"
            axs[col].set_title(state_number)
            xl = [RADIUS, RADIUS + RADIUS * probability * np.cos(phase + pi/2)]
            yl = [RADIUS, RADIUS + RADIUS * probability * np.sin(phase + pi/2)]
            axs[col].plot(xl, yl, 'r')
            axs[col].axis('off')
        plt.show()