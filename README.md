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


