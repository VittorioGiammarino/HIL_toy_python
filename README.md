# HIL_toy_python
Code to solve first a stochastic shortest path problem by means of value iteration.
Then data are sampled from this optimal solution and Behavioral cloning and Hierarchical Imitation Learning are used to infer the optimal policy.

# Main
- BC_main.py  (for Behavioral Cloning)
- HIL_main (for Hierarchical Imitation Learning)
- main (for regularizers validation and comparison between the two methods)

## BC_main.py

### Dependencies
```python
import tensorflow
from tensorflow import keras
import numpy
import matplotlib.pyplot
import matplotlib.animation
```

### Pipeline
- Generate Expert's policy
- Sample data from Expert
- Train Neural Networks for BC
  - NN1 uses Cross Entropy loss function
  - NN2 Mean Squared Error
  - NN3 Hinge Loss
- Evaluate Performance of NNs with different number of trajectories




