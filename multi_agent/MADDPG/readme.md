# MADDPG

MADDPG implementation. (I started from the [paper](https://arxiv.org/pdf/1706.02275.pdf) and this youtube [video](https://youtu.be/tZTQ6S9PfkE?si=RWU-qpIMj0fCUEXM))

## File org

* *argParser.py*: contains the argument parser for the program
* *MADDPG.py*: contains the MADDPG class, which implements the MADDPG algorithm
* *ma_replay_buffer.py*: contains the MultiAgentReplayBuffer class, which implements a replay buffer for the MADDPG algorithm, each agent has its own replay buffer
* *main.py*: contains, env setup, MADDPG setup, training loop.

## Experiments

The code has been tested on [MPE envs](https://pettingzoo.farama.org/environments/mpe/), in particular on simple_speaker_listener_v4, simple_spreading_v4, simple_adversary_v3.

We tested on few episodes (10k), just to see if the algorithm works on the envs, as the img shows.
![Speaker listener hp](image.png)

To reach a good reward we have to increase the number of episodes, i.e. for simple spread we need to reach 100k episodes.

### Simple adversary case
In this enviroment, that is competitive and no cooperative for reach a good reward, increase the number of episodes its not enough, we need to make the enviroment more "stable"/"stationary" so we need to generalize it, to do that we make the agent able to learn different policies for the same task (sub policies chapter in the original paper).
You can change this parameter with the *argParser*.

# Notes on implementation
* The code is able to manage envs with different number of action/obs between the agents.