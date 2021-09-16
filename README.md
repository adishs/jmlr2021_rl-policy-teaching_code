## "Policy Teaching in Reinforcement Learning via Environment Poisoning Attacks", JMLR 2021

## Text from JMLR Release form
THIS SOURCE CODE IS SUPPLIED “AS IS” WITHOUT WARRANTY OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF MACHINE LEARNING RESEARCH (JMLR) AND JMLR’S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER ASSUMES ALL LIABILITY AND RESPONSIBILITY FOR USE OF THIS SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR JMLR’S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR DAMAGES OF ANY KIND RESULTING FROM ITS USE. Without limiting the generality of the foregoing, neither the author, nor JMLR, nor JMLR’s publishers and distributors, warrant that the Source Code will be error-free, will operate without interruption, or will meet the needs of the user.

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
