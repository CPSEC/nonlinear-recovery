import numpy as np
from simulators.nonlinear.continuous_stirred_tank_reactor import CSTR

# simulation settings
max_index = 1000
dt = 0.02
ref = [np.array([0, 300])] * (max_index + 1)
noise = {
    'process': {
        'type': 'box_uniform',
        'param': {'lo': np.array([-1e-3, -1e-1]), 'up': np.array([1e-3, 1e-1])}
    }
}
# noise = None
cstr_model = CSTR('test', dt, max_index, noise)


# control loops
for i in range(0, max_index + 1):
    assert cstr_model.cur_index == i
    cstr_model.update_current_ref(ref[i])
    # attack here
    cstr_model.evolve()


# plot results
import matplotlib.pyplot as plt

t_arr = np.linspace(0, 10, max_index + 1)
ref = [x[1] for x in cstr_model.refs[:max_index + 1]]
y_arr = [x[1] for x in cstr_model.outputs[:max_index + 1]]

plt.plot(t_arr, y_arr, t_arr, ref)
plt.show()

u_arr = [x[0] for x in cstr_model.inputs[:max_index + 1]]
plt.plot(t_arr, u_arr)
plt.show()