from scipy.optimize import approx_fprime
import numpy as np
from functools import partial

class Linearizer:
    def __init__(self, ode, nx, nu):
        self.ode = ode
        self.nx = nx
        self.nu = nu

    def at(self, x_0: np.ndarray, u_0: np.ndarray):
        # ode(t, x, u)
        A = approx_fprime(x_0, lambda x: self.ode(0, x, u_0))
        B = approx_fprime(u_0, lambda u: self.ode(0, x_0, u))
        assert A.shape == (self.nx, self.nx)
        assert B.shape == (self.nx, self.nu)
        c = self.ode(0, x_0, u_0) - A@x_0 - B@u_0
        return A, B, c

if __name__ == '__main__':
    from simulators.nonlinear.continuous_stirred_tank_reactor import cstr
    linearize = Linearizer(cstr, nx=2, nu=1)
    x_0 = np.array([1, 300])
    u_0 = np.array([280])
    A, B, c = linearize.at(x_0, u_0)
    print(f'{A=}, \n{B=}, \n{c=}')

