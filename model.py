import numpy as np


gK = 36.0  # (mS/cm^2)
gNa = 120.0  # (mS/cm^2)
gL = 0.3  # (mS/cm^2)

Cm = 1.0  # (uF/cm^2)

VK = -77.0  # EK(mV)
VNa = 50.0  # EK(mV)
Vl = -54.387  # EK(mV)


def alpha_n(Vm):
    return 0.01 * (Vm + 55.0) / (1.0 - np.exp(-0.1 * (Vm + 55.0)))


def beta_n(Vm):
    return 0.125 * np.exp(-0.0125 * (Vm + 65.0))


def alpha_m(Vm):
    return 0.1 * (Vm + 40.0) / (1.0 - np.exp(-0.1 * (Vm + 40.0)))


def beta_m(Vm):
    return 4.0 * np.exp(-0.0556 * (Vm + 65.0))


def alpha_h(Vm):
    return 0.07 * np.exp(-0.05 * (Vm + 65.0))


def beta_h(Vm):
    return 1.0 / (1.0 + np.exp(-0.1 * (Vm + 35.0)))


def n_0(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))


def m_0(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))


def h_0(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


def generate_computing_derivatives_function(input_stimulus_func):
    def compute_derivatives(y, t0):
        dy = np.zeros((4,))

        Vm = y[0]
        n = y[1]
        m = y[2]
        h = y[3]

        # dVm/dt
        GK = (gK / Cm) * np.power(n, 4.0)
        GNa = (gNa / Cm) * np.power(m, 3.0) * h
        GL = gL / Cm

        dy[0] = (input_stimulus_func(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))

        # dn/dt
        dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)

        # dm/dt
        dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)

        # dh/dt
        dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)

        return dy
    return compute_derivatives
