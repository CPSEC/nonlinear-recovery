# Real-Time Data-Predictive Attack-Recovery for Complex Cyber-Physical Systems

This repository contains the code for our RTAS submission.

## Setup
Install required python packages.
```
pip install -r requirements.txt
```

## Running the code for cstr, quadrotor, and vessel
```
cd rtas
python compare_all.py --sim cstr_bias
python compare_all.py --sim quad_bias
python compare_all.py --sim vessel_bias
```

Plots will be produced and saved inside rtas/figs folder.

## Information

* The compare_all.py file compares all recovery methods and saves a plot in rtas/figs folder.
* The mpc_only.py file only executes our proposed data-predictive recovery algorithm and saves a plot in the same folder.
* The setting for each bias attack experiment can be found in rtas/settings.py.
* the simulators can be found in nonlinear-recovery/simulators/nonlinear

