import gymnasium as gym
import torch
from TD3 import TD3, make_env
from DDPG import DDPG

import collections
import numpy as np
import wandb

from argParser import parse_args
# 
# from torchrl import envs


args = parse_args()

device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
env_name='LunarLander-v2'
args.env_id = env_name
#torch.set_default_dtype(torch.float64)
# env = gym.vector.SyncVectorEnv([make_env(env_name, seed=args.seed)])
# assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"

# env.single_observation_space.dtype = np.float32
config_DDPG = {'name': 'DDPG', 'alg': DDPG}
config_TD3 = {'name': 'TD3', 'alg': TD3}
project_name = f"Compare DDPG vs TD3, env {env_name}"
for s in [2]:
    args.seed = s
    env = gym.vector.SyncVectorEnv([make_env(env_name, seed=args.seed)])
    assert isinstance(env.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    env.single_observation_space.dtype = np.float32
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    for test in [config_TD3]:
        # if test['name'] == 'TD3' and s == 1: continue 
        # if test['name'] == 'DDPG' and (s == 2 or s == 1) : continue
        # wandb.init(
        #     project=project_name,
        #     name=f"{test['name']} {args.seed}",
        #     group=test['name'], 
        #     job_type=env_name,
        #     reinit=True
        # )
        agent = test['alg'](args, env)
        reward_mean = agent.training_loop()
        torch.save(agent.actor.state_dict(), f'models/{test["name"]}/actor_{int(reward_mean)}_{args.env_id}_{args.seed}.pth')
        print("Run: ", test["name"], " Reaward last 100: ", reward_mean)