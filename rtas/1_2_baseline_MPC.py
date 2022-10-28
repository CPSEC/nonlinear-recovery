from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append('../')

from settings import quadruple_tank_bias
from utils.controllers.MPC_cvxpy import MPC
from utils.observers.full_state_bound import Estimator

# ready exp: lane_keeping,
exps = [quadruple_tank_bias]
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
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    est = Estimator(A, B, max_k=150, epsilon=1e-7)

    recovery_complete_index = np.inf # init to big value
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index - 1:
            print('normal_state=', exp.model.cur_x)
        if exp.recovery_index <= i < recovery_complete_index:
            # x_0 estimation
            us = exp.model.inputs[exp.attack_start_index:i]
            x_0 = exp.model.states[exp.attack_start_index]
            x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
            print(f'{exp.attack_start_index=},{exp.recovery_index=},\n{us=}')
            print(f'{x_cur_lo=},\n {x_cur_up=},\n {exp.model.states[i]=}')

            # deadline estimate only once
            if i == exp.recovery_index:
                safe_set_lo = exp.safe_set_lo
                safe_set_up = exp.safe_set_up
                control = exp.model.inputs[i-1]
                k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100 )
                print(f'deadline {k=}')
                recovery_complete_index = exp.attack_start_index + k

            # get recovery control sequence
            mpc_settings = {
                'Ad': A, 'Bd': B,
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
            print(f'{x_cur=},{rec_u=}')

            u = rec_u[0]
            exp.model.evolve(u)
            print(f'{i=}, {exp.model.cur_x=}')
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
    # recovery_complete_index = result['w/o']['recovery_complete_index']
    t_arr = np.linspace(0, exp.dt * recovery_complete_index, recovery_complete_index + 1)
    # y_arr = [x[exp.output_index] for x in result['w/o']['outputs'][:recovery_complete_index + 1]]
    ref = [x[exp.ref_index] for x in exp.model.refs[:recovery_complete_index + 1]]
    plt.plot(t_arr, ref, color='black', linestyle='dashed')
    # plt.plot(t_arr, y_arr, label='w/o')

    recovery_complete_index = result['w/']['recovery_complete_index']
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
    plt.show()



