from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

import sys
sys.path.append('../')

from settings import cstr_bias, quad_bias
from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
from utils.controllers.MPC_cvxpy import MPC
from utils.observers.full_state_bound_nonlinear import NonlinearEstimator
from utils.control.linearizer import Linearizer, analytical_linearize_cstr

# ready exp: lane_keeping,
exps = [quad_bias] # [cstr_bias]
result = {}   # for print or plot
for exp in exps:

    # ----------------------------------- w/ kalman filter ---------------------------
    exp.model.reset()
    print('=' * 20, exp.name, '=' * 20)
    k = None
    rec_u = None
    recovery_complete_index = None
    x_cur_predict = None
    x_cur_update = None
    cin = None

    u_lo = exp.model.controller.control_lo
    u_up = exp.model.controller.control_up
    linearize = Linearizer(ode=exp.model.ode, nx=exp.nx, nu=exp.nu, dt=exp.dt)
    est = NonlinearEstimator(exp.ode_imath, exp.dt)

    recovery_complete_index = np.inf # init to big value
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)

        if i == exp.attack_start_index - 1:
            print('normal_state=', exp.model.cur_x)
        if exp.recovery_index <= i < recovery_complete_index:

            print(f'{i=:-^40}')

            # x_0 estimation
            us = exp.model.inputs[exp.attack_start_index:i]
            xs = exp.model.states[exp.attack_start_index:i+1]
            x_0 = exp.model.states[exp.attack_start_index]
            print(f'estimating x_cur now...')
            x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us, xs, exp.unsafe_states_onehot)
            print(f'{exp.attack_start_index=},{exp.recovery_index=}')
            print(f'{x_cur_lo=},\n {x_cur_up=},\n {exp.model.states[i]=}')

            # deadline estimate only once
            if i == exp.recovery_index:
                safe_set_lo = exp.safe_set_lo
                safe_set_up = exp.safe_set_up
                control = exp.model.inputs[i-1]
                print(f'estimating deadline now...')
                k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, max_k=100)
                # k=35
                print(f'deadline {k=}')
                recovery_complete_index = exp.attack_start_index + k

            # Linearize and Discretize
            print(f'{i=},{x_cur=},x_gt={exp.model.cur_x}')
            # Ad, Bd, cd = analytical_linearize_cstr(x_cur, exp.model.inputs[i-1], exp.dt)  # (2, 2) (2, 1) (2,)
            Ad, Bd, cd = linearize.at(x_cur, exp.model.inputs[i-1])  

            # get recovery control sequence
            mpc_settings = {
                'Ad': Ad, 'Bd': Bd, 'c_nonlinear': cd,
                'Q': exp.Q, 'QN': exp.QN, 'R': exp.R,
                'N': k+3-(i-exp.recovery_index), # maintainable time=3! # receding horizon MPC!
                'ddl': k-(i-exp.recovery_index), 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                'control_lo': exp.control_lo, 'control_up': exp.control_up,
                'ref': exp.recovery_ref
            }
            mpc = MPC(mpc_settings)
            _ = mpc.update(feedback_value=x_cur)
            rec_u = mpc.get_full_ctrl()
            # print(f'{x_cur=},{rec_u=}')

            u = rec_u[0]
            print(f'{u=}')
            exp.model.evolve(u)
            print(f'after evolve - {exp.model.cur_x=}')
        else:
            if i == recovery_complete_index:
                print('state after recovery:', exp.model.cur_x)
                step = recovery_complete_index-exp.recovery_index
                print(f'use {step} steps to recover.')
            exp.model.evolve()

    result['w/'] = {}
    result['w/']['outputs'] = deepcopy(exp.model.outputs)
    result['w/']['recovery_complete_index'] = recovery_complete_index

    # plot
    plt.rcParams.update({'font.size': 18})  # front size
    fig = plt.figure(figsize=(8, 4))
    plt.title(exp.name+' y_'+str(exp.output_index))
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    plt.plot(t_arr, ref, color='black', linestyle='dashed')

    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    y_arr = [x[exp.output_index] for x in result['w/']['outputs'][:recovery_complete_index + 1]]
    plt.plot(t_arr, y_arr, label='w/')

    plt.vlines(exp.attack_start_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='red', linestyle='dashed')
    plt.vlines(exp.recovery_index*exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2.5)
    plt.hlines(exp.strip[0], exp.x_lim[0], exp.x_lim[1], colors='grey')
    plt.hlines(exp.strip[1], exp.x_lim[0], exp.x_lim[1], colors='grey')
    plt.ylim(exp.y_lim)
    plt.xlim(exp.x_lim)
    plt.legend(loc='best')
    os.makedirs('./fig', exist_ok=True)
    plt.savefig('./fig/'+exp.name+'_MPC.png', format='png', bbox_inches='tight')
    # plt.show()



