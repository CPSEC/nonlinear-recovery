import numpy as np

import sys
sys.path.append('../')

from simulators.linear.motor_speed import MotorSpeed
from simulators.linear.quadruple_tank import QuadrupleTank
from simulators.nonlinear.continuous_stirred_tank_reactor import CSTR, cstr_imath
from simulators.nonlinear.quad import quadrotor, quad_imath, quad, quad_jfx
from simulators.nonlinear.inverted_pendulum import InvertedPendulum, inverted_pendulum_imath
from simulators.nonlinear.vessel import VESSEL, deg2rad, heading_circle, kn2ms, whole_model_imath
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
    nx = 2
    nu = 1

    # plot
    y_lim = (280, 360)
    x_lim = (8, dt * 200)
    strip = (target_set_lo[output_index], target_set_up[output_index])
    y_label = 'Temperature [K]'

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([274.57786])
    x_ss = np.array([0.98472896, 300.00335862])

#---------------quadrotor----------------------------
class quad_bias:
    name = 'quad_bias'
    max_index = 500
    ref = [np.array([0,0,0,0,0,0,0,0,5,0,0,0])] * (max_index+1)
    dt = 0.01
    noise_term = 0.00024
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'up': np.array([noise_term, noise_term, noise_term, noise_term, noise_term, noise_term, noise_term, noise_term, 0.0001, noise_term, noise_term, 0.0001])} # change 0.01 to 1 or 5 or something
        }
    }
    # noise = None
    model = quadrotor(name, dt, max_index, noise=noise)
    ode_imath = quad_imath
    
    attack_start_index = 250 # index in time
    recovery_index = 270 # index in time
    bias = np.array([0, 0, 0, 0, 0 ,0, 0, 0, -1, 0, 0, 0])
    unsafe_states_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    attack = Attack('bias', bias, attack_start_index)
    
    output_index = 8 # index in state
    ref_index = 8 # index in state

    target_set_lo = np.array([-1e20, -1e20, -1e20, -1e20, -1e20, -1e20, -1e20, -1e20, 5-0.2, -1e20, -1e20, -1e20])
    target_set_up = np.array([1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 5+0.2, 1e20, 1e20, 1e20])
    safe_set_lo = np.array([-1e20, -1e20, -1e20, -1e20, -1e20, -1e20, -1e20, -1e20, 0, -1e20, -1e20, -1e20])
    safe_set_up = np.array([1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 200, 1e20, 1e20, 1e20])


    control_lo = np.array([-10])
    control_up = np.array([100])
    recovery_ref = np.array([0,0,0,0,0,0,0,0,5,0,0,0])

    # Q = np.diag([1, 1])
    # QN = np.diag([1, 1])
    Q = np.eye(12)
    Q[8, 8] = 100000
    QN = np.eye(12)
    QN[8, 8] = 100000
    R = np.diag([10])

    MPC_freq = 10
    nx = 12
    nu = 1

    # plot
    y_lim = (4, 7)
    x_lim = (2.4, 4)
    y_label = 'Altitude (M)'
    strip = (target_set_lo[output_index], target_set_up[output_index])

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([9.81])
    x_ss = np.array([0,0,0,0,0,0,0,0,5,0,0,0])

    # for EKF
    C_during_atk = np.diag([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    f = lambda x, u, dt: x + dt * quad(None, x, u)
    jh = lambda x, u, dt: np.diag([0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    StateCov = np.eye(12)*(0.00001)
    SensorCov = np.eye(12)*(0.00001)
    jf = quad_jfx
    

    # -------------------- Inverted Pendulum ----------------------------
class pendulum_bias:
    name = 'pendulum_bias'
    max_index = 800
    ref = [np.array([1, 0, np.pi, 0])]  * (max_index+1)
    dt = 0.02
    # noise = {
    #     'process': {
    #         'type': 'box_uniform',
    #         'param': {'lo': np.array([0, 0, 0, 0]), 'up': np.array([0.0001, 0.0001, 0.0001, 0.0001])} # change 0.01 to 1 or 5 or something
    #     }
    # }
    noise = None
    model = InvertedPendulum(name, dt, max_index, noise=noise)
    ode_imath = inverted_pendulum_imath
    
    attack_start_index = 300 # index in time
    recovery_index = 330 # index in time
    bias = np.array([0, 0, 0.01, 0])
    unsafe_states_onehot = [0, 0, 1, 0]
    attack = Attack('bias', bias, attack_start_index)
    final = np.array([0, 0, np.pi, 0])
    output_index = 2 # index in state
    ref_index = 2 # index in state
    safe_set_lo = [-100000, -100000, np.pi-2, -100000]
    safe_set_up = [100000, 100000, np.pi+2, 100000]
    target_set_lo = np.array([-100000, -100000, np.pi-2, -100000])
    target_set_up = np.array([100000, 100000, np.pi+2, 100000])
    control_lo = np.array([-50])
    control_up = np.array([50])
    recovery_ref = np.array([1, 0, np.pi, 0])

    Q = np.eye(4)
    Q[0, 0] = 100
    QN = np.eye(4)
    QN[0, 0] = 100
    R = np.diag([10])

    MPC_freq = 1
    nx = 4
    nu = 1

    # plot
    y_lim = (2.5, 4)
    x_lim = (5.5, dt * 700)
    strip = (target_set_lo[output_index], target_set_up[output_index])
    y_label = 'Angle (Rad)'

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([0])
    x_ss = np.array([1, 0, 3.14, 0])

    #---------------vessel----------------------------
class vessel_bias:
    name = 'vessel_bias'
    max_index = 500
    ref = [np.array([0, 0, heading_circle(deg2rad(90)), kn2ms(1), 0,0,0,0])] * (max_index+1)
    dt = 1
    noise_term = 0.015
    noise = {
        'process': {
            'type': 'box_uniform',
            'param': {'lo': np.array([0, 0, 0, 0, 0, 0, 0, 0]), 'up': np.array([noise_term, noise_term, noise_term, 0.0001, noise_term, noise_term, noise_term, noise_term])} # change 0.01 to 1 or 5 or something
        }
    }
    # noise = None
    model = VESSEL(name, dt, max_index, noise=noise)
    ode_imath = whole_model_imath
    
    attack_start_index = 300 # index in time
    recovery_index = 310 # index in time
    bias = np.array([0, 0, 0, -0.5, 0 ,0, 0, 0])
    unsafe_states_onehot = [0, 0, 0, 1, 0, 0, 0, 0]
    attack = Attack('bias', bias, attack_start_index)
    
    output_index = 3 # index in state
    ref_index = 3 # index in state

    target_set_lo = np.array([-1e20, -1e20, -1e20, 0.5, -1e20, -1e20, -1e20, -1e20])
    target_set_up = np.array([1e20, 1e20, 1e20, 0.9, 1e20, 1e20, 1e20, 1e20])
    safe_set_lo = np.array([-1e20, -1e20, -1e20, -1500, -1e20, -1e20, -1e20, -1e20])
    safe_set_up = np.array([1e20, 1e20, 1e20, 1500, 1e20, 1e20, 1e20, 1e20])


    control_lo = np.array([0, -1])
    control_up = np.array([2, 1])
    recovery_ref = np.array([0, 0, heading_circle(deg2rad(90)), kn2ms(1), 0,0,0,0])

    # Q = np.diag([1, 1])
    # QN = np.diag([1, 1])
    Q = np.eye(8)
    Q[3, 3] = 10000000
    QN = np.eye(8)
    QN[3, 3] = 10000000
    R = np.diag([10, 10])

    MPC_freq = 1
    nx = 8
    nu = 2

    # plot
    y_lim = (0, 2.5)
    x_lim = (295, 500)
    y_label = 'Speed(m/s)'
    strip = (target_set_lo[output_index], target_set_up[output_index])

    # for linearizations for baselines, find equilibrium point and use below
    u_ss = np.array([0.1,0])
    x_ss = np.array([0, 0, heading_circle(deg2rad(90)), kn2ms(1), 0,0,0,0])