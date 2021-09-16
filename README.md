JMLR 2021 -- Policy Teaching in Reinforcement Learning via Environment Poisoning Attacks

## Prerequisites
```
Python3
Matplotlib
Numpy
Scipy
Cvxpy
Itertools
```

## Running the code
To get results, you will need to go either averaged/ or discounted/ directory and run the following scripts:

## For the Chain environment

### For online attack
```
python teaching_online.py
```

### For offline attacks when varying parameter $\overline{R}(s_0, .)$
```
python teaching_offline_vary_c.py
```

### For offline attack when varying parameter $\epsilon$
```
python teaching_offline_vary_epsilon.py
```

### To see how long it takes to solve different attack problems when |S|=4, |S|=10, |S|=20, |S|=30, |S|=50, |S|=70 and |S|=100, you should go to the averaged/ directory and run:
```
python teaching_time_table.py
```
## ==========================================

## For the Gridworld environment

### For online attack 
```
python teaching_online_grid.py
```

### For offline attacks when varying parameter $\overline{R}(s_0, .)$ 
```
python teaching_offline_vary_c_grid.py
```

### For offline attack when varying parameter $\epsilon$
```
python teaching_offline_vary_epsilon_grid.py
```

### Results

After running the above scripts, new plots will be created in plots/env_chain or in plots/env_grid directory accordingly.

In the __main__ function, the variable number_of_iterations denotes the number of runs used to average the results. Set a smaller number for faster execution.
