import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

from model import n_0, m_0, h_0, generate_computing_derivatives_function

# Set random seed (for reproducibility)
np.random.seed(30)


def plot_results(Idv, Vy, T, plot_phase_space=False):
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(T, Idv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'Current density (uA/$cm^2$)')
    ax.set_title('Stimulus (Current density)')
    plt.grid()
    fig.show()

    # Neuron potential
    fig, ax = plt.subplots(figsize=(24, 14))
    ax.plot(T, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential')
    plt.grid()
    fig.show()

    # n, m, h
    fig, ax = plt.subplots(figsize=(24, 14))
    for i, label in enumerate(['n', 'm', 'h']):
        ax.plot(T, Vy[:, i + 1], label=label)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potassium & Sodium ion-channel rates')
    ax.set_title('Potassium & Sodium ion-channel rates in time')
    ax.legend()
    plt.grid()
    fig.show()

    # Trajectories with limit cycles
    if plot_phase_space:
        for i, label in enumerate(['Vm - n', 'Vm - m', 'Vm - h']):
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.plot(Vy[:, 0], Vy[:, i + 1], label=label)
            ax.set_title('Limit cycles({})'.format(label))
            ax.legend()
            plt.grid()
            fig.show()


def run_simulation(tmin, tmax, Id, plot_phase_space=False):
    T = np.linspace(tmin, tmax, 10000)

    Vm0 = -65.0
    Y = np.array([Vm0, n_0(Vm0), m_0(Vm0), h_0(Vm0)])

    compute_derivatives = generate_computing_derivatives_function(Id)

    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    Vy = odeint(compute_derivatives, Y, T)

    # Input stimulus
    Idv = [Id(t) for t in T]

    plot_results(Idv, Vy, T, plot_phase_space)


if __name__ == '__main__':
    ################################################################################
    #######################################    1   #################################
    tmin = -50.0
    tmax = 480.0

    def Id3(t):
        if 0.0 < t < 30.0:
            return 20
        if 80.0 < t < 130.0:
            return 20
        if 180.0 < t < 280.0:
            return 32
        if 330.0 < t < 430.0:
            return 47
        return 0.0

    # run_simulation(tmin, tmax, Id3, plot_phase_space=True)

    ################################################################################
    #######################################    2   #################################

    # Start and end time (in milliseconds)
    tmin = -5.0
    tmax = 200.0

    # Input stimulus
    def Id2(t):
        if t < 0.0:
            # to check if system in equilibrium when simulation starts
            return 0.0
        if np.ceil(t) % 4 == 1:
            return 150.0
        if np.ceil(t) % 4 == 3:
            return 100.0
        return 0.0

    #run_simulation(tmin, tmax, Id2, plot_phase_space=False)



