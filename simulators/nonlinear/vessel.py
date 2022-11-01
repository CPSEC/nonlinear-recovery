import numpy as np
from utils import Simulator
from utils.controllers.PID import PID
from math import cos, sin, copysign, pi
from interval import imath, interval
#Helper Function
def kn2ms(x):
    return 1852*x/3600.0

def ms2kn(x):
    return 3600*x/1852.0

def nmi2m(x):
    return 1852*x

def m2nmi(x):
    return x/1852.0

def deg2rad(x):
    return x*pi/180

def rad2deg(x):
    return x*180/pi

def saturate(x, a, b, use_imath=False):
    if use_imath:
        if b < x[0][0]:
            min_of_b_and_x = interval([b, b])
        else:
            min_of_b_and_x = x

        if a > min_of_b_and_x[0][1]:
            max_of_a_and_above = interval([a, a])
        else:
            max_of_a_and_above = x 

        return max_of_a_and_above

        # return imath.max(a, imath.min(x, b))
    else:
        return max(a, min(x, b))

def heading_circle(x):
    if x > pi:
        return x-2*pi
    elif x < -pi:
        return x+2*pi
    return x
#Initial state
x_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# control parameters
Kp1 = 4.5e-1 # 3.0
Ki1 = 5e-2#1/30
Kd1 = 0
Kp2 = 1e-1# 2.7
Ki2 = 0
Kd2 = -3.5e0



control_limit = {
    'lo': np.array([0, -1]),
    'up': np.array([1, 1])
}
# print(heading_circle(deg2rad(45)))
#ODE
def vessel(x, u, use_imath=False):
    if use_imath:
        cs, ss = imath.cos(x[2]), imath.sin(x[2])
    else:
        cs, ss = cos(x[2]), sin(x[2])
    #parameters
    m = 127.92     # Mass, kg
    Xg = 0     # X coordinate for center of gravity, m
    Iz = 61.967     # Moment of inertia for Z axis (i.e., heave axis), kg.m^2
	    
    # Linear damping parameters
    # Xu = 0
    # Yv = 0
    Yr = 0
    Nv = 0
    # Nr = 0
    Xu = -2.332
    Yv = -4.673
    Nr = -0.01675


	# Inertia matrix parameters

    # Xu_dot = 0
    # Yv_dot = 0
    # Yr_dot = 0
    # Nv_dot = 0
    # Nr_dot = 0
    Xu_dot = 3.262
    Yv_dot = 28.89
    Yr_dot = 0.525
    Nv_dot = 0.157
    Nr_dot = 13.98
	# Restorative forces parameters
    # Xx = 0
    # Yy = 0
    # Npsi = 0
    # Helper variables
    a1 = Iz*Yv_dot - Nr_dot*Yv_dot + Nv_dot*Yr_dot - Iz*m + Nr_dot*m + (Xg*m)**2 - Nv_dot*Xg*m - Xg*Yr_dot*m
    a2 = u[1] + Yr*x[2] * Yv*x[1]
    a3 = u[2] + Nr*x[2] * Nv*x[1]

    # State transition functions
    f1 = x[3]*cs - x[4]*ss
    f2 = x[3]*ss + x[4]*cs
    f3 = x[5]
    f4 = (u[0] + Xu*x[3])/(m-Xu_dot)
    f5 = (1/a1)*( (Xg*m - Yr_dot)*a3 + (Nr_dot - Iz)*a2 )
    f6 = (1/a1)*( (m - Yv_dot)*a3 + (m*Xg - Nv_dot)*a2 )

    # States derivative
    return [f1, f2, f3, f4, f5, f6]

def rudder(command, surge_v, use_imath=False):
    rudder_v = 0.166
    rudder_r = 1.661
    rudder_max_angle = deg2rad(30)
    angle = saturate(command, -rudder_max_angle, rudder_max_angle, use_imath)
    return -rudder_v * surge_v * angle, -rudder_r * surge_v * angle

def copysign_custom(a, b, use_imath=False):
    if use_imath:
        return copysign(a, b[0][0])
    else:
        return copysign(a, b)

def propulsion(torque, shaft_speed, use_imath=False):
    Is = 25000.0/747225.0    # Moment of inertia
    Kq = 0.0113   # Load torque coefficient
    Kt = 0.3763   # Load thrust coefficient
    D = 0.3
    max_shaft_speed = 157
    rho = 1000
    this_shaft = (1/Is)*( torque - copysign_custom(1, shaft_speed, use_imath)*Kq*rho*D**5*(shaft_speed/(2*pi))**2 )
    omega = saturate(shaft_speed, -max_shaft_speed, max_shaft_speed, use_imath)
    return copysign_custom(1, omega, use_imath) * Kt * rho * D**4 * (omega/(2*pi))**2, this_shaft

def dcmotor(power, current, omega, use_imath=False):
    K = 0.3                         # Torque and back EMF constant
    R = 1                         # Armature resistance
    L = 0.5                         # Armature inductance
    efficiency = 0.8      # Motor efficiency (i.e., yield)
    max_current = 22
    this_current = (power- R*current - K*omega)/L
    I = saturate(current, -max_current, max_current, use_imath)
    return (K*I)*efficiency, this_current

def battery(signal):
	voltage = 12
	return voltage*signal


def whole_model(t, x, u, use_imath=False):
	# state 0-5 for vessel (East, North, Yaw, and their velocities)
	# state 6 for propeller Angular shaft speed
	# state 7 for motor current

	# two inputs: The first is for the battery, the second is for the rudder command

    power = battery(u[0])

    torque, current = dcmotor(power, x[7], x[6], use_imath)
    thrust, shaft = propulsion(torque, x[6],use_imath)
    f_sway, f_yaw = rudder(u[1], x[3], use_imath)
    first_6 = vessel(x[:6], [thrust, f_sway, f_yaw], use_imath)
    if use_imath:
        first_6.append(shaft)
        first_6.append(current)
    else:
        first_6.append(shaft)
        first_6.append(current)
        first_6 = np.array(first_6)
    return first_6
def whole_model_imath(t,x,u):
    return whole_model(t=t,x=x,u=u,use_imath=True)




class Controller:
    def __init__(self, dt):
        self.dt = dt
        self.pid1 = PID(Kp1, Ki1, Kd1, current_time=-dt)
        self.pid1.setWindup(100)
        self.pid1.setSampleTime(dt)
        self.pid2 = PID(Kp2, Ki2, Kd2, current_time=-dt)
        self.pid2.setWindup(100)
        self.pid2.setSampleTime(dt)
        self.set_control_limit(control_limit['lo'], control_limit['up'])
        self.pid2.setWindup(deg2rad(30))

    def update(self, ref: np.ndarray, feedback_value: np.ndarray, current_time) -> np.ndarray:
        self.pid1.set_reference(ref[3])
        cin1 = self.pid1.update(feedback_value[3], current_time)
        self.pid2.set_reference(ref[2])
        cin2 = self.pid2.update(feedback_value[2], current_time)
        return np.array([cin1, cin2])

    def set_control_limit(self, control_lo, control_up):
        self.control_lo = control_lo
        self.control_up = control_up
        self.pid1.set_control_limit(control_lo[0], control_up[0])
        self.pid2.set_control_limit(control_lo[1], control_up[1])

    def clear(self):
        self.pid1.clear(-self.dt)
        self.pid2.clear(-self.dt)





class VESSEL(Simulator):
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
        super().__init__('Vessel ' + name, dt, max_index)
        self.nonlinear(ode=whole_model, n=8, m=2, p=8)  # number of states, control inputs, outputs
        controller = Controller(dt)
        settings = {
            'init_state': x_0,
            'feedback_type': 'state',  # 'state' or 'output',  you must define C if 'output'
            'controller': controller
        }
        if noise:
            settings['noise'] = noise
        self.sim_init(settings)


if __name__ == "__main__":
    max_index = 500
    dt = 1
    ref = [np.array([0, 0, heading_circle(deg2rad(90)), kn2ms(1), 0,0,0,0])] * (max_index+1)
    # ref = [np.array([0, 0, 45, 1, 0,0,0,0])] * (max_index+1)
    # bias attack example
    from utils import Attack
    bias = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    bias_attack = Attack('bias', bias, 300)
    ip = VESSEL('test', dt, max_index)
    for i in range(0, max_index + 1):
        assert ip.cur_index == i
        ip.update_current_ref(ref[i])
        # attack here
        ip.cur_feedback = bias_attack.launch(ip.cur_feedback, ip.cur_index, ip.states)
        ip.evolve()

    # print results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(6, 1)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax
    t_arr = np.linspace(0, 5, max_index + 1)
    ref1 = [x[3] for x in ip.refs[:max_index + 1]]
    y1_arr = [x[3] for x in ip.outputs[:max_index + 1]]
    ax1.set_title('Speed')
    ax1.plot(t_arr, y1_arr, t_arr, ref1)
    ref2 = [x[2] for x in ip.refs[:max_index + 1]]
    y2_arr = [x[2] for x in ip.outputs[:max_index + 1]]
    ax2.set_title('Yaw')
    ax2.plot(t_arr, y2_arr, t_arr, ref2)
    y3_arr = [x[1] for x in ip.outputs[:max_index + 1]]
    ax3.set_title('East')
    ax3.plot(t_arr, y3_arr)
    y4_arr = [x[0] for x in ip.outputs[:max_index + 1]]
    ax4.set_title('Position')
    ax4.plot(y3_arr, y4_arr)
    y5_arr = [x[1] for x in ip.inputs[:max_index + 1]]
    ax5.set_title('Inputs')
    ax5.plot(t_arr, y5_arr)
    y6_arr = [x[0] for x in ip.inputs[:max_index + 1]]
    ax6.set_title('Inputs')
    ax6.plot(t_arr, y6_arr)
    plt.show()