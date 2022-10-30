import numpy as np

import sys
sys.path.append('../')

from simulators.linear.motor_speed import MotorSpeed
from simulators.linear.quadruple_tank import QuadrupleTank
from simulators.nonlinear.continuous_stirred_tank_reactor import CSTR, cstr_imath
from utils.attack import Attack
from utils.formal.strip import Strip


# --------------------- motor speed -------------------
class motor_speed_bias:
    # needed by 0_attack_no_recovery
    name = 'motor_speed_bias'
    max_index = 500
    dt = 0.02
    ref = [np.array([4])] * 501
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.array([[0.01, 0], [0, 0.04]])}
        }
    }
    model = MotorSpeed('bias', dt, max_index, noise)
    control_lo = np.array([0])
    control_up = np.array([60])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 150
    bias = np.array([-1])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 180

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0]), a=-4.3, b=-3.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (2.8, 4.2)
    y_lim = (3.65, 5.2)
    y_label = 'Rotational speed [rad/sec]'
    strip = (4.2, 3.8)

    kf_C = np.array([[0, 1]])
    k_given = 40  # new added
    kf_R = np.diag([1e-7])

    # baseline
    safe_set_lo = np.array([4, 30])
    safe_set_up = np.array([5.07, 80])
    target_set_lo = np.array([3.9, 35.81277766])
    target_set_up = np.array([4.1, 60])
    recovery_ref = np.array([4,  41.81277766])
    Q = np.diag([100, 1])
    QN = np.diag([100, 1])
    R = np.diag([1])


# -------------------- quadruple tank ----------------------------
class quadruple_tank_bias:
    # needed by 0_attack_no_recovery
    name = 'quadruple_tank_bias'
    max_index = 300
    dt = 1
    ref = [np.array([7, 7])] * 1001 + [np.array([7, 7])] * 1000
    noise = {
        'process': {
            'type': 'white',
            'param': {'C': np.diag([0.02, 0.02, 0.02, 0.02])}
        }
    }
    model = QuadrupleTank('test', dt, max_index, noise)
    control_lo = np.array([0, 0])
    control_up = np.array([10, 10])
    model.controller.set_control_limit(control_lo, control_up)
    attack_start_index = 150
    bias = np.array([-2.0, 0])
    attack = Attack('bias', bias, attack_start_index)
    recovery_index = 160

    # needed by 1_recovery_given_p
    s = Strip(np.array([-1, 0, 0, 0]), a=-14.3, b=-13.7)
    P_given = 0.95
    max_recovery_step = 40
    # plot
    ref_index = 0
    output_index = 0
    x_lim = (140, 250)
    y_lim = (2.7, 9)
    y_label = 'Water level [cm]'
    strip = (7.15, 6.85)  # modify according to strip

    kf_C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # depend on attack
    k_given = 40
    kf_R = np.diag([1e-7, 1e-7, 1e-7])

    # baseline
    safe_set_lo = np.array([0, 0, 0, 0])
    safe_set_up = np.array([20, 20, 20, 20])
    target_set_lo = np.array([13.8, 13.8, 0, 0])
    target_set_up = np.array([14.2, 14.2, 20, 20])
    recovery_ref = np.array([14, 14, 2, 2.5])
    Q = np.diag([1, 1, 0, 0])
    QN = np.diag([1, 1, 0, 0])
    R = np.diag([1, 1])


# -------------------- cstr ----------------------------
class cstr_bias:
    name = 'cstr_bias'
    max_index = 160
    ref = [np.array([0.98189, 300.00013])] * (max_index+1)
    dt = 0.1
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.array([0, 0]), 'up': np.array([0.1, 0.1])} # change 0.01 to 1 or 5 or something
        }
    }
    # noise = None
    model = CSTR(name, dt, max_index, noise=noise)
    ode_imath = cstr_imath
    
    attack_start_index = 100 # index in time
    recovery_index = 110 # index in time
    bias = np.array([0, -30])
    unsafe_states_onehot = [0, 1]
    attack = Attack('bias', bias, attack_start_index)
    
    output_index = 1 # index in state
    ref_index = 1 # index in state

    safe_set_lo = np.array([-5, 250])
    safe_set_up = np.array([5, 360])
    target_set_lo = np.array([-5, 299])
    target_set_up = np.array([5, 301])
    control_lo = np.array([250])
    control_up = np.array([350])
    recovery_ref = np.array([0.98189, 300.00013])

    # Q = np.diag([1, 1])
    # QN = np.diag([1, 1])
    Q = np.diag([1, 1000])
    QN = np.diag([1, 1000])
    R = np.diag([1])

    MPC_freq = 1

    # plot
    y_lim = (280, 360)
    x_lim = (8, dt * 200)
    strip = (target_set_lo[output_index], target_set_up[output_index])
    y_label = 'Temperature [K]'

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([274.57786])
    x_ss = np.array([0.98472896, 300.00335862])


