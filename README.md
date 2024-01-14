# One-Shot Reachability Analysis of Neural Network Dynamical Systems: Is it Worth Verifying a Rollout Dynamics? 

<img src="https://github.com/ShaoruChen/web-materials/blob/main/One_shot_ICRA_23/illustration_3.png" width="600" height="250">

For a potentially uncertain NN dynamical system $x_{t+1} = f(x_t, w_t)$, it is often required to bound the reachable sets of it over a finite horizon given an initial set $x_0 \in X_0$ and bounded disturbance set $w_t \in W$. With the computed reachable set over-approximations, we can certify the reach-avoid properties of the NN system as shown in the figure above. The closed-loop NN dynamics may consist of various interconnected NN and uncertainty modules shown below:

<img src="https://github.com/ShaoruChen/web-materials/blob/main/One_shot_ICRA_23/nnds.png" width="600" height="250">

and the goal is to obtain reachable set over-approximations $R_t$ of $x_t$ for $t= 0, 1, \cdots, T$ as tight as possible. Essentially, this can be reduced to a NN verification problem, but **the selection of the computation graph matters**. This repository compares two frameworks of conducting reachability analysis:

1. **Recursive Analysis**: Derive reachable set over-approximation $R_{t+1}$ with $R_t$ as the input domain to the NN dynamics  $x_{t+1} = f(x_t, w_t)$. 
2. **One-Shot Analysis**: Derive reachable set over-approximation $R_{t+1}$ with the initial set $X_0$ as the input domain to the rollout NN dynamics $x_{t+1} = f^{(t)}(x_0, w_0, w_1, \cdots, w_t)$.

<p float="left">
<img src="https://github.com/ShaoruChen/web-materials/blob/main/One_shot_ICRA_23/duffing_backward.png" width="450" height="400">
<img src="https://github.com/ShaoruChen/web-materials/blob/main/One_shot_ICRA_23/duffing_forward.png" width="450" height="400">
</p>

## Reference
This repository compares the **one-shot** and **recursive** reachability analysis frameworks for discrete-time NN dynamical systems over a finite horizon. It contains examples from the paper:

[One-shot reachability analysis of neural network dynamical systems](https://arxiv.org/pdf/2209.11827.pdf) \
Shaoru Chen, Victor M. Preciado, Mahyar Fazlyab \
2023 IEEE International Conference on Robotics and Automation (ICRA)

## Installation and Running Examples
The following installation has been tested:

```
conda create -n nn_reach python=3.10
conda activate nn_reach

# Example: install pytorch on macOS. 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install auto-lirpa
conda install -c conda-forge cvxpy
conda install scipy
pip install -r requirements.txt
```

To generate Fig.3.a, run
```
python examples/duffing_oscillator/duffing_oscillator_propagation.py --method backward --one_shot_horizon 8 --recursive_horizon 3
```

To generate Fig.3.b, run
```
python examples/duffing_oscillator/duffing_oscillator_propagation.py --method forward --one_shot_horizon 3 --recursive_horizon 2
```

To generate Fig.4, run
```
python examples/duffing_oscillator/duffing_oscillator_LP.py
```

To generate Fig.6, run
```
python examples/cartpole/cartpole_propagation.py
```


## Third-party dependence
We use [auto-LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) to implement the bound propagation method to bound NN outputs.

We use [pympc](https://github.com/TobiaMarcucci/pympc/tree/master) mainly for operations on polyhedron.

