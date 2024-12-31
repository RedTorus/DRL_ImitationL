import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from modules import PolicyNet, ValueNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from PIL import Image

try:
    import wandb
except ImportError:
    wandb = None

mean_rewards = []
median_rewards = []
max_rewards = []

class TrainDaggerBC:

    def __init__(self, env, model, expert_model, optimizer, states, actions, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        model.set_device(self.device)
        self.expert_model = self.expert_model.to(self.device)

        self.mode = mode

        if self.mode == "BC":
            self.states = []
            self.actions = []
            self.timesteps = []
            for trajectory in range(states.shape[0]):
                trajectory_mask = states[trajectory].sum(axis=1) != 0
                self.states.append(states[trajectory][trajectory_mask])
                self.actions.append(actions[trajectory][trajectory_mask])
                self.timesteps.append(np.arange(0, len(trajectory_mask)))
            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.clip_sample_range = 1
            self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        else:
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
        """

        states, old_actions, timesteps, rewards = [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()  
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                p = policy(torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0), torch.tensor(t).to(self.device).long().unsqueeze(0))
            a = p.cpu().numpy()[0]
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state

        return states, old_actions, timesteps, rewards

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(state_tensor, deterministic=True).cpu().numpy()
            action = np.clip(action, -1, 1)[0]
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards=[] 
        for i in range(num_trajectories_per_batch_collection):
            states, actions, timesteps, reward = self.generate_trajectory(self.env, self.model)
            #print("length of states: ", len(states))
            exper_action = self.call_expert_policy(states)
            #print("Trajectory: ", i)
    
            if self.states is None:
                self.states = states
            else:
                self.states = np.concatenate((self.states, states), axis=0)

            if self.actions is None:
                self.actions = exper_action
            else:
                self.actions = np.concatenate((self.actions, exper_action), axis=0)
            
            if self.timesteps is None:
                self.timesteps = timesteps
            else:
                self.timesteps = np.concatenate((self.timesteps, timesteps), axis=0)

            rewards.append(np.sum(reward))   

        # END STUDENT SOLUTION

        return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards=[]
        for i in range(num_trajectories_per_batch_collection):
            states, actions, timesteps, reward = self.generate_trajectory(self.env, self.model)

            # self.states=np.concatenate((self.states, states), axis=0)
            # self.actions=np.concatenate((self.actions, actions), axis=0)
            # self.timesteps=np.concatenate((self.timesteps, timesteps), axis=0)
            rewards.append(np.sum(reward))

        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps, 
        num_BC_training_steps=20000,
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_BC_training_steps: the number of iterations to train the model using BC.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION
        losses=[]
        evaluate_rewards_every=1000
        total_loss=0
        if self.mode == "BC":
            for i in tqdm(range(num_BC_training_steps)):

                if wandb_logging:
                    wandb.init(
                        name="BC training",
                        group="BC",
                        project='walker deepRL HW3',
                    )
                
                

                loss=self.training_step(batch_size)
                #total_loss+=loss

                #total_loss=total_loss/(num_training_steps_per_batch_collection*num_batch_collection_steps)
                losses.append(loss)
                if wandb_logging:
                    wandb.log({"loss": loss})

                if (i%print_every ==0 ):
                    print("Loss: ", loss)

                if (i%evaluate_rewards_every ==0):
                    rewards=self.generate_trajectories(num_trajectories_per_batch_collection)
                    mean_reward= np.mean(rewards)
                    median_reward= np.median(rewards)
                    max_reward= np.max(rewards)
                    mean_rewards.append(mean_reward)
                    median_rewards.append(median_reward)
                    max_rewards.append(max_reward)

                #if (i%save_every ==0):
        if self.mode == "DAgger":

            if wandb_logging:
                    wandb.init(
                        name="DAgger training",
                        group="DAgger",
                        project='walker deepRL HW3',
                    )

            for i in tqdm(range(num_batch_collection_steps)):
                #print("Update Training Data")
                rewards=self.update_training_data(num_trajectories_per_batch_collection)
                #print("Training data updated")
                for j in range(num_training_steps_per_batch_collection):
                    #print("Training Step: ", j)
                    loss=self.training_step(batch_size)
                    #total_loss+=loss
                    losses.append(loss)
                    if wandb_logging:
                        wandb.log({"loss": loss})
                #losses.append(total_loss/num_training_steps_per_batch_collection)
                #print("Loss appended")
                if (i%print_every ==0 ):
                    print("Loss: ", loss)
                

                if (i%1 ==0):
                    mean_reward= np.mean(rewards)
                    median_reward= np.median(rewards)
                    max_reward= np.max(rewards)
                    mean_rewards.append(mean_reward)
                    median_rewards.append(median_reward)
                    max_rewards.append(max_reward)
                #print("Loop iteration done")
                #if (i%save_every ==0):
                
        if wandb_logging:
            wandb.finish()
        # END STUDENT SOLUTION

        return losses

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad()
        predicted_actions = self.model(states, timesteps)
        loss = loss_fn(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        # num_samples = len(self.states)
        # if batch_size > num_samples:
        #     batch_size = num_samples
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
            
        
        return states, actions, timesteps

def run_training():
    """
    Simple Run Training Function
    """

    env = gym.make('BipedalWalker-v3') # , render_mode="rgb_array"
    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "super_expert_PPO_model"
    expert_model = PolicyNet(24, 4)
    model_weights = torch.load(f"data/models/{model_name}.pt", map_location=device)
    expert_model.load_state_dict(model_weights["PolicyNet"])
    mode = "DAgger"
    states_path = f"data/states_BC.pkl"
    actions_path = f"data/actions_BC.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    # BEGIN STUDENT SOLUTION
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = SimpleNet(state_dim, action_dim, 128, max_episode_length=1600)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    trainer = TrainDaggerBC(env, model, expert_model, optimizer, states, actions, device=device,mode=mode)
    losses = trainer.train(20, num_BC_training_steps=20000, num_training_steps_per_batch_collection=1000, num_trajectories_per_batch_collection=20, batch_size=64, print_every=500, save_every=10000, wandb_logging=False)

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Losses")
    plt.show()
    print("final loss value: ", losses[-1])
    print("Mean Rewards length: ", len(mean_rewards))
    fig, ax = plt.subplots()
    ax.plot(mean_rewards)
    ax.plot(median_rewards)
    ax.plot(max_rewards)
    ax.set_title("Rewards")
    ax.legend(["Mean", "Median", "Max"])
    plt.show()

    create_gif(trainer)


    # END STUDENT SOLUTION

def create_gif(Agent: TrainDaggerBC):

    gif_path = f"data/{Agent.mode}.gif"
    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    frames=[]
    state, _ = env.reset()
    done=False
    for i in range(1600):
        with torch.no_grad():
            p = Agent.model(torch.tensor(state).float().unsqueeze(0), torch.tensor(i).long().unsqueeze(0))[0].cpu().numpy()
        next_state, reward, done, trunc, _ = env.step(p)
        frame = env.render()
        frames.append(Image.fromarray(frame))

        if done:
            break
        state = next_state

    env.close()
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=30)  
    print("Gif saved in path: BipedalWalker-v3.gif")


if __name__ == "__main__":
    run_training()
    