# Experiments Notes

This note is used to document the parameter tuning process for taxi rebalancing project.
It is necessary to keep track of the methods we have tried and parameters we have tuned
for future reference. 

## Current Method
* **SAC**: we started with this method. It is complicated and has no good convergence guarantee.
* **DQN**: and its variants. Classic method simple to tune. We need to reformulate
our action to discrete space for this method.
* **DDPG**: good for continuous control problem.
* **PPO**: state of the art policy gradient method.

## Parameters to tune
### SAC
* Learning rates for all three networks.
* Policy update rate tau
* NN structure
* Temperature coefficient alpha
* Target Q network update frequency
* Gradient norm clipping
* experience replay
* Episode length
* Batch size

#### 4/24/2020
* Start with a fixed temperature coefficient (alpha = 1) 
* 5 nodes with 80 cars at start
* Gradient norm clipping at 5
* Total iteration 3000 iter times 5 episodes per iter times 60 steps per episodes
* Default nn config (256, 256) for both
* Today's experiments:
    1. a smaller actor net (128, 128), (128, ), (256, ), (256, 128), (128, 256)
    2. default
    3. with experience replay
    4. double episode length