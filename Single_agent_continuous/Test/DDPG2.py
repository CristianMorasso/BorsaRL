# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
# DDPG (Deep Deterministic Policy Gradient) is a type of Actor-Critic method that uses an actor
# network to represent the policy and a critic network for value estimation. The
# algorithm learns by updating both networks simultaneously using gradient ascent on the
# combined reward/value function, which is defined as the sum of rewards plus some penalty
# term. This approach has been successful in solving continuous control tasks such as
# reaching target locations or manipulating objects.

from distutils.util import strtobool
import wandb
import collections
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

def make_env(env_id, seed=1):
    def thunk():
        env = gym.make(env_id)#continuous=True
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_mu = nn.Linear(32, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class DDPG():
    """Interacts with and learns from the environment."""
    def __init__(self, args, env, writer=None):
        self.args = args
        self.env = env
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        #Definitions
        self.replay_buffer = ReplayBuffer(args.buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space,
            args.device,
            handle_timeout_termination=False,
        )
        self.actor = Actor(env).to(args.device)
        self.actor.double()
        self.target_actor = Actor(env).to(args.device)
        self.target_actor.double()
        self.qf1 = QNetwork(env).to(args.device)
        self.qf1.double()
        self.qf1_target = QNetwork(env).to(args.device)
        self.qf1_target.double()

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=args.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.learning_rate)

        
    def select_action(self, obs, noise_mul):
        actions = self.actor(torch.Tensor(obs).double().to(self.args.device))#.double()
        noise = torch.normal(0, self.actor.action_scale * noise_mul )
        actions += noise
        actions = actions.cpu().numpy().clip(self.env.single_action_space.low, self.env.single_action_space.high)
        return actions
    
    def update(self):
        for _ in range(20):
            data = self.replay_buffer.sample(self.args.batch_size)


            with torch.no_grad():
                next_state_actions = self.target_actor(data.next_observations)
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            self.q_optimizer.zero_grad()
            qf1_loss.backward()
            self.q_optimizer.step()

        #if ep % 2 == 0:
        actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the target network
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


    def training_loop(self):
        success_th = 70
        reward_queue =  collections.deque(maxlen=100)
        done_queue =  collections.deque(maxlen=100)
        for ep in range(self.args.n_ep):
            obs, _ = self.env.reset(seed=self.args.seed+ep)
            # obs.dtype = np.float32
            #obs = np.float32(obs)
            for _ in range(self.args.n_step):
                # ALGO LOGIC: put action logic here
            
                with torch.no_grad():
                    ep_temp = ep/(self.args.n_ep/5000)
                    self.noise_mul = (98.5/100)**((ep_temp)/7.5)#(self.args.n_ep-ep)/self.args.n_ep
                    actions = self.select_action(obs, self.noise_mul)
                    

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

                
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        

                        real_next_obs = next_obs.copy()
                        for idx, trunc in enumerate(truncations):
                            if trunc:
                                real_next_obs[idx] = infos["final_observation"][idx]
                        #rb.add(obs, actions, rewards, real_next_obs,  terminations )
                        self.replay_buffer.add(obs,real_next_obs, actions, rewards,   terminations, infos)
                        break
                    break
            
                self.replay_buffer.add(obs,next_obs,actions,rewards,   terminations, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
            info= infos["final_info"][0]
            reward_queue.append(info['episode']['r'])
            done_queue.append(terminations)
            if ep % 250 == 0 :
                print(f"ep={ep}, episodic_return={float(info['episode']['r']):.2}, mean={float(np.mean(reward_queue)):.2}, success_rate={np.mean(done_queue)}")
            wandb.log({'moving_100_rwg':np.mean(reward_queue),'ep': ep,'noise_mul': self.noise_mul,'ep_length':info["episode"]["l"], 'success_rate':np.mean(done_queue)} )
            # if np.mean(done_queue) > success_th: 
            #     torch.save(self.actor.state_dict(), f'models/DDPG/actor_{int(np.mean(done_queue))}_{self.args.env_id}_{self.args.seed}.pth')
            #     success_th = np.mean(done_queue)

            self.update()
        return np.mean(reward_queue)