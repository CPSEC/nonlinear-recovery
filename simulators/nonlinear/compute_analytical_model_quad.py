import sympy as sym
import numpy as np
import math
from copy import deepcopy

# ----------------------------------------------------
Ix, Iy, Iz, mass, g, phi, theta, psi, w_phi, w_theta, w_psi, x, y, z, v_x, v_y, v_z, U_t, U_phi, U_theta, U_psi  = sym.symbols('Ix Iy Iz mass g phi theta psi w_phi w_theta w_psi x y z v_x v_y v_z U_t U_phi U_theta U_psi')

vars = [
    phi,
    theta,
    psi,
    w_phi,
    w_theta,
    w_psi,
    x,
    y,
    z,
    v_x,
    v_y,
    v_z,
]

us = [
    U_phi,
    U_theta,
    U_psi,
    U_t,
]

f = [
    w_phi, 
    w_theta, 
    w_psi, 
    U_phi/Ix + w_theta * w_psi * (Iy - Iz)/Ix, 
    U_theta/Iy + w_phi * w_psi * (Iz - Iz)/Iy, 
    U_psi/Iz + w_phi * w_theta * (Ix - Iy)/Iz, 
    v_x,
    v_y,
    v_z,
    U_t/mass * (sym.cos(phi)* sym.sin(theta) * sym.cos(psi) + sym.sin(phi) * sym.sin(psi)),
    U_t/mass * (sym.cos(phi)* sym.sin(theta) * sym.sin(psi) - sym.sin(phi) * sym.cos(psi)),
    U_t/mass * sym.cos(phi) * sym.cos(theta) - g,
]
# ----------------------------------------------------

dt = sym.symbols('dt')
J = sym.zeros(len(f), len(vars))
for i, fi in enumerate(f):
    for j, s in enumerate(vars):
        J[i, j] = sym.diff(fi, s)
Ac = deepcopy(J)
Ad = sym.eye(len(vars)) + dt * Ac
print(f'{Ad=}')

# J = sym.zeros(len(f), len(us))
# for i, fi in enumerate(f):
#     for j, s in enumerate(us):
#         J[i, j] = sym.diff(fi, s)
# Bc = deepcopy(J)
# Bd = dt * Bc
# print(f'{Bd=}')




