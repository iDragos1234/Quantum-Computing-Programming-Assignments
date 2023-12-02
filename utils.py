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
            state_number = "|" + str(state_index) + "⟩"
            axs[col].set_title(state_number)
            xl = [RADIUS, RADIUS + RADIUS * probability * np.cos(phase + pi/2)]
            yl = [RADIUS, RADIUS + RADIUS * probability * np.sin(phase + pi/2)]
            axs[col].plot(xl, yl, 'r')
            axs[col].axis('off')
        plt.show()

def plot_experiment(counts):
    n_states = len(counts)
    zeros = []
    ones = []

    for k in counts:
        zeros.append(k['0'])
        ones.append(k['1'])

    remapped_counts = {
        '0': np.array(zeros) / 100,
        '1': np.array(ones) / 100,
    }

    x = np.arange(n_states)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    plt.rcParams['figure.figsize'] = [12, 5]
    fig, ax = plt.subplots(layout='constrained')

    for out_state, probs in remapped_counts.items():
        offset = (0.1 + width) * multiplier + 0.075
        rects = ax.bar(x + offset, probs, width, label=f'Prob(|{out_state}⟩)')
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Input state')
    ax.set_ylabel('Output state probability, Prob(|out_state⟩)')
    ax.set_title('Probabilities of output states in percentages')
    ax.set_xticks(x + width, [f'|{k}⟩' for k in range(n_states)])
    ax.legend(loc='upper left')
    # ax.set_ylim(0, 250)

    plt.show()