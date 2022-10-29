from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

import sys
sys.path.append('../')

from settings import cstr_bias
from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
from utils.controllers.MPC_cvxpy import MPC
from utils.observers.full_state_bound_nonlinear import NonlinearEstimator
from utils.control.linearizer import Linearizer, analytical_linearize

# ready exp: lane_keeping,
exps = [cstr_bias]
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
    linearize = Linearizer(ode=exp.model.ode, nx=2, nu=1, dt=exp.dt)
    est = NonlinearEstimator(exp.ode_imath, exp.dt)

    recovery_complete_index = np.inf # init to big value
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        exp.model.evolve()

    result['w/'] = {}
    result['w/']['outputs'] = deepcopy(exp.model.outputs)
    result['w/']['recovery_complete_index'] = recovery_complete_index

    # plot
    recovery_complete_index = exp.max_index
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
    plt.savefig('./fig/'+exp.name+'_no_recovery.png', format='png', bbox_inches='tight')
    # plt.show()



