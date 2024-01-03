## Single Agent Algorithms
Here there are float32 and float64 implementation, choose  the one that fit your environment needs.

To run you enviroment you have to change the `env_name` in `main_continuous.py`, it will run DDPG and TD3 with 3 different seeds (1,2,3).

You can change the number of episodes with `--n_ep num_episodes`, from the comand line.

You can change the noise multiplier function in `main_continuous.py`.

Use Weights and Biases to log the results.