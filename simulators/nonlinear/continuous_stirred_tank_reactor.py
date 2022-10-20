
import numpy as np
from utils import Simulator
from utils.controllers.PID import PID

# Parameters:
# Volumetric Flowrate (m^3/sec)
q = 100
# Volume of CSTR (m^3)
V = 100
# Density of A-B Mixture (kg/m^3)
rho = 1000
# Heat capacity of A-B Mixture (J/kg-K)
Cp = 0.239
# Heat of reaction for A->B (J/mol)
mdelH = 5e4
# E - Activation energy in the Arrhenius Equation (J/mol)
# R - Universal Gas Constant = 8.31451 J/mol-K
EoverR = 8750
# Pre-exponential factor (1/sec)
k0 = 7.2e10
# U - Overall Heat Transfer Coefficient (W/m^2-K)
# A - Area - this value is specific for the U calculation (m^2)
UA = 5e4

def cstr(x, t, u, Tf=350, Caf=1):
    # Inputs (3):
    # Temperature of cooling jacket (K)
    Tc = u
    # Tf = Feed Temperature (K)
    # Caf = Feed Concentration (mol/m^3)

    # States (2):
    # Concentration of A in CSTR (mol/m^3)
    Ca = x[0]
    # Temperature in CSTR (K)
    T = x[1]

    # reaction rate
    rA = k0 * np.exp(-EoverR / T) * Ca

    # Calculate concentration derivative
    dCadt = q / V * (Caf - Ca) - rA
    # Calculate temperature derivative
    dTdt = q / V * (Tf - T) \
           + mdelH / (rho * Cp) * rA \
           + UA / V / rho / Cp * (Tc - T)

    # Return xdot:
    xdot = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot


# initial states
x_0 = np.array([-1, 0, np.pi + 0.1, 0])

# control parameters
KP = 0.5 * 1.0
KI = KP / (3 / 8.0)
KD = - KP * 0.1
control_limit = {
    'lo': np.array([250]),
    'up': np.array([350])
}


class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.pid = PID(KP, KI, KD, current_time=-dt)
        self.pid.setWindup(100)
        self.pid.setSampleTime(dt)
        self.set_control_limit(control_limit['lo'], control_limit['up'])

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid.set_reference(ref[0])
        cin = self.pid.update(feedback_value[1], current_time)      # only use the 2nd state here
        return np.array([cin])

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.pid.set_control_limit(self.control_lo[0], self.control_up[0])

    def clear(self):
        self.pid.clear(current_time=-self.dt)


class CSTR(Simulator):
    """
        States: (2,)
            x[0]: Concentration of A in CSTR (mol/m^3)
            x[1]: Temperature in CSTR (K)
        Control Input: (1,)
            u[0]: Temperature of cooling jacket (K)
        Output:  (2,)
            State Feedback
        Controller: PID
    """
    def __init__(self, name, dt, max_index, noise=None):
        super().__init__('CSTR ' + name, dt, max_index)
        self.nonlinear(ode=cstr, n=2, m=1, p=2)    # number of states, control inputs, outputs
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'state',     # 'state' or 'output',  you must define C if 'output'
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)



