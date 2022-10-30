import scipy.optimize as opt
import numpy as np
import sys
sys.path.append('../')
from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
from utils.control.linearizer import Linearizer, analytical_linearize_cstr

def cstr_with_fixed_control(x):
    u_ss = [274.57786]
    return cstr(t=None, x=x, u=u_ss)

if __name__ == "__main__":
    u_ss = [274.57786]
    x_ss = opt.fsolve(cstr_with_fixed_control, (0.98189, 300.00013)) 
    print(f'{x_ss=}')
    print(f'{cstr_with_fixed_control(x_ss)=} \n')

    linearize = Linearizer(ode=cstr, nx=2, nu=1, dt=0.1)
    A, B, c = linearize.at(x_ss, u_ss)  
    print(f'{A=}')
    print(f'{B=}')
    print(f'{c=}')