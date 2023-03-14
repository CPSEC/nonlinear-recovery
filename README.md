# Real-Time Data-Predictive Attack-Recovery for Complex Cyber-Physical Systems

This repository contains the code for our RTAS 2023 paper.

## Setup
Install required python packages.
```
pip install -r requirements.txt
```

## Running the code
For cstr, quadrotor, and vessel reocvery. The below will reproduce the plots in Figure 8 and save them in rtas/fig.
```
cd rtas
python compare_all.py --sim cstr_bias
python compare_all.py --sim quad_bias
python compare_all.py --sim vessel_bias
```

For quadrotor recovery with observer (Figure 12; right).
```
python compare_all_including_obs.py
```

For overhead box plots (Figure 13).
```
python compare_all_including_obs_and_overhead.py
```

For sensitivity analysis, change the settings in each class in rtas/settings.py. Rerun the above lines to reproduce the plots for the new settings.

## Information

* The compare_all.py file compares all recovery methods and saves a plot in rtas/figs folder.
* The mpc_only.py file only executes our proposed data-predictive recovery algorithm and saves a plot in the same folder.
* The setting for each bias attack experiment can be found in rtas/settings.py.
* the simulators can be found in nonlinear-recovery/simulators/nonlinear

