import gymnasium as gym
import os
import numpy as np
import torch
from torch import nn

from tqdm import tqdm
import pickle

from diffusion_policy_transformer import PolicyDiffusionTransformer
from PIL import Image
from diffusers import DDPMScheduler, DDIMScheduler
import time
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

class TrainDiffusionPolicy:

    def __init__(
        self,
        env,
        model, 
        optimizer, 
        states_array, 
        actions_array, 
        device="cpu", 
        num_train_diffusion_timesteps=30,
        max_trajectory_length=1600,
    ):
        """
        Initializes the TrainDiffusionPolicy class. Creates necessary data structures and normalizes states AND actions.

        Args:
            env (gym.Env): The environment that the model is trained on.
            model (PolicyDiffusionTransformer): the model to train
            optimizer (torch.optim.Optimizer): the optimizer to use for training the model
            states_array (np.ndarray): the states to train on
            actions_array (np.ndarray): the actions to train on
            device (str): the device to use for training
            num_train_diffusion_timesteps (int): the number of diffusion timesteps to use for training
        """
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.states = states_array
        self.actions = actions_array

        self.action_dimension = self.actions.shape[-1]
        self.state_dimension = self.states.shape[-1]

        # clip all actions to be between -1 and 1, as this is the range that the environment expects
        self.clip_sample_range = 1
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        self.trajectory_lengths = [sum(1 for s in self.states[i] if np.sum(s) != 0) for i in range(len(self.states))]
        self.max_trajectory_length = max_trajectory_length

        model.set_device(self.device)

        # normalize states and actions
        all_states = np.concatenate([self.states[i, 0:self.trajectory_lengths[i]] for i in range(len(self.states))], axis=0)
        all_actions = np.concatenate([self.actions[i, 0:self.trajectory_lengths[i]] for i in range(len(self.actions))], axis=0)

        self.states_mean = np.mean(all_states, axis=(0))
        self.states_std = np.std(all_states, axis=(0))
        self.states = (self.states - self.states_mean) / self.states_std

        self.actions_mean = np.mean(all_actions, axis=(0))
        self.actions_std = np.std(all_actions, axis=(0))
        self.actions = (self.actions - self.actions_mean) / self.actions_std

        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps

        # training and inference schedulers for diffusion
        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log", # variance is different for inference, see paper https://arxiv.org/pdf/2301.10677
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = self.inference_scheduler.alphas_cumprod.to(self.device)


    def get_inference_timesteps(self):
        """
        gets the timesteps to use for inference
        """
        self.inference_scheduler.set_timesteps(self.num_train_diffusion_timesteps, device=self.device)
        return self.inference_scheduler.timesteps

    def diffusion_sample(
        self,
        previous_states, 
        previous_actions,
        episode_timesteps,
        previous_states_padding_mask=None,
        previous_actions_padding_mask=None,
        actions_padding_mask=None,
        max_action_len=3,
    ):
        """
        perform a single diffusion sample from noise to actions

        Args:
            previous_states (torch.Tensor): the previous states to condition on
            previous_actions (torch.Tensor): the previous actions to condition on
            episode_timesteps (torch.Tensor): the episode timesteps to condition on
            previous_states_padding_mask (torch.Tensor): the padding mask for the previous states
            previous_actions_padding_mask (torch.Tensor): the padding mask for the previous actions
            actions_padding_mask (torch.Tensor): the padding mask for the actions being predicted
            max_action_len (int): the maximum number of actions to predict

        NOTE: remember that you are predicting max_action_len actions, not just one
        """
        # BEGIN STUDENT SOLUTION
        batch_size = previous_states.shape[0]
        xT = torch.randn((batch_size, max_action_len, self.action_dimension)).to(self.device)
        timesteps= self.get_inference_timesteps()
        
        for t in timesteps:

            noise_levels = torch.full((batch_size, 1), t, device=self.device)
            #z = torch.randn(batch_size, device=self.device) if t>1 else torch.zeros(batch_size, device=self.device)
            predictions = self.model(
                previous_states.type(torch.float),
                previous_actions.type(torch.float),
                xT,
                episode_timesteps.type(torch.long),
                noise_levels,
                previous_states_padding_mask,
                previous_actions_padding_mask,
                actions_padding_mask,    
            )

            xnew = self.inference_scheduler.step(predictions, noise_levels, xT).prev_sample
            xT = xnew

        # END STUDENT SOLUTION
        predicted_actions = xT
        return predicted_actions

    def sample_trajectory(
        self, 
        env, 
        num_actions_to_eval_in_a_row=3, 
        num_previous_states=5,
        num_previous_actions=4, 
    ):
        """
        run a trajectory using the trained model

        Args:
            env (gym.Env): the environment to run the trajectory in
            num_actions_to_eval_in_a_row (int): the number of actions to evaluate in a row
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on

        NOTE: use with torch.no_grad(): to speed up inference by not storing gradients
        NOTE: for the first few steps, make sure to add padding to previous states/actions - use False if a state/action should be included, and True if it should be padded
        NOTE: both states and actions should be normalized before being passed to the model, and the model outputs normalized actions that need to be denormalized
        NOTE: refer to the forward function of diffusion_policy_transformer to see how to pass in the inputs (tensor shapes, etc.)
        """
        # BEGIN STUDENT SOLUTION
        statesize = env.observation_space.shape[0]
        actionsize = env.action_space.shape[0]
        s,a,t,done, trunc = [env.reset()[0]], [], [0], False, False
        s[0] = (s[0] - self.states_mean) / self.states_std
        for i in range(num_previous_states-1):
            s.append((np.zeros(statesize)-self.states_mean)/self.states_std)
            t.append(0)

        for i in range(num_previous_actions):
            a.append((np.zeros(actionsize)-self.actions_mean)/self.actions_std)

        #s= (s - self.states_mean) / self.states_std
        #a= (a - self.actions_mean) / self.actions_std

        count2=0
        count = 0
        prev_states= torch.zeros((num_previous_states, self.state_dimension), device=self.device)
        prev_actions= torch.zeros((num_previous_actions, self.action_dimension), device=self.device)
        rewards=np.zeros(1600)
        frames=[]
        while not done and not trunc:
            
            # if count < num_previous_states:
            #     state_padding_mask = torch.tensor([True] * (num_previous_states - count) + [True] * count).unsqueeze(0).to(self.device)
            # else:
            #     state_padding_mask = torch.tensor([True] * num_previous_states).unsqueeze(0).to(self.device)

            # if count < num_previous_actions:
            #     action_padding_mask = torch.tensor([True] * (num_previous_actions - count) + [True] * count).unsqueeze(0).to(self.device)
            # else:
            #     action_padding_mask = torch.tensor([True] * num_previous_actions).unsqueeze(0).to(self.device)
            #state_padding_mask = torch.tensor([True] * max(0, num_previous_states - count) + [False] * min(count, num_previous_states)).unsqueeze(0).to(self.device)
            # action_padding_mask = torch.tensor([True] * max(0, num_previous_actions - count) + [False] * min(count, num_previous_actions)).unsqueeze(0).to(self.device)
            state_padding_mask = torch.tensor([True] * (num_previous_states - min(count, num_previous_states)) + [False] * min(count, num_previous_states), dtype=bool).unsqueeze(0).to(self.device)
            #state_padding_mask= state_padding_mask.bool()
            action_padding_mask = torch.tensor([True] * (num_previous_actions - min(count, num_previous_actions)) + [False] * min(count, num_previous_actions), dtype=bool).unsqueeze(0).to(self.device)
            #action_padding_mask= action_padding_mask.bool()
            #print("state_padding_mask", state_padding_mask)
            
            if count == 0:
                s_norm =s
                a_norm =a
                td= t
            else:
                s_norm = s#[::-1]
                a_norm = a#[::-1]
                td= t#[::-1]
            #s_norm = s#(s - self.states_mean) / self.states_std
            s_norm= np.stack(s_norm, axis=0)
            s_norm= torch.from_numpy(s_norm).float().to(self.device)
            #a_norm = a #(a - self.actions_mean) / self.actions_std
            a_norm= np.stack(a_norm, axis=0)
            a_norm= torch.from_numpy(a_norm).float().to(self.device)
            td= torch.tensor(td).unsqueeze(0).to(self.device)
            count += 1
            with torch.no_grad():
                actions = self.diffusion_sample(
                    s_norm.unsqueeze(0).float(), 
                    a_norm.unsqueeze(0), 
                    td, 
                    max_action_len=num_actions_to_eval_in_a_row, 
                    previous_states_padding_mask=state_padding_mask, 
                    previous_actions_padding_mask=action_padding_mask, 
                    actions_padding_mask=None)
            actions = actions * torch.tensor(self.actions_std, device=self.device) + torch.tensor(self.actions_mean, device=self.device)
            actions = actions.squeeze(0).cpu().detach().numpy()
            for i in range(num_actions_to_eval_in_a_row):
                snew, r, done, trunc, _ = env.step(actions[i])
                rewards[count2] = r
                
                frame= env.render()
                frames.append(Image.fromarray(frame))
                if done or trunc:
                    break
                snew = (snew - self.states_mean) / self.states_std
                actionnew= (actions[i] - self.actions_mean) / self.actions_std
                #print("len of s", len(s))
                #print("snew", snew)
                s.append(snew)
                a.append(actionnew)
                t.append(count2+1)
                count2 += 1
            
            s,a,t = s[-num_previous_states:], a[-num_previous_actions:], t[-num_previous_states:]
            #print("sis       ", s)

        gif_path = f"data/diffusion.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=30)
        # END STUDENT SOLUTION
        return rewards

    




    def evaluation(
        self,
        diffusion_policy_iter=None, 
        num_samples=20,
        num_actions_to_eval_in_a_row=3,
    ):
        """
        evaluate the model on the environment

        Args:
            diffusion_policy_iter (Optional[int]): the iteration to load the diffusion policy from
            num_samples (int): the number of samples to evaluate

        NOTE: feel free to change this function when making graphs
        """
        # load model weights:
        if diffusion_policy_iter is None:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy.pt", map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{diffusion_policy_iter}.pt", map_location=self.device))

 
        self.model.eval() # turn on eval mode (this turns off dropout, running_mean, etc. that are used in training)

        rewards = np.zeros((num_samples, self.max_trajectory_length))
        os.makedirs("data/diffusion_policy_trajectories", exist_ok=True)
        for sample_trajectory in tqdm(range(num_samples)):
            time1 = time.time()
            reward = self.sample_trajectory(self.env, num_actions_to_eval_in_a_row=num_actions_to_eval_in_a_row)
            time2 = time.time()
            print(f"trajectory {sample_trajectory} took {time2 - time1} seconds")
            rewards[sample_trajectory] = reward
            print(f"rewards from trajectory {sample_trajectory}={reward.sum()}")
        print(f"average reward per trajectory={rewards.sum() / (rewards.shape[0])}")
        print(f"median reward per trajectory={np.median(rewards.sum(axis=1))}")
        print(f"max reward per trajectory={np.max(rewards.sum(axis=1))}")
        print(f"average trajectory length={np.mean(np.array([sum(1 for r in rewards[i] if r != 0) for i in range(len(rewards))]))}")

    def train(
        self, 
        num_training_steps, 
        batch_size=64, 
        print_every=5000, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        training loop that calls training_step

        Args:
            num_training_steps (int): the number of training steps to run
            batch_size (int): the batch size to use
            print_every (int): how often to print the loss
            save_every (int): how often to save the model
            wandb_logging (bool): whether to log to wandb
        """
        model = self.model
        if wandb_logging:
            wandb.init(
                name="diffusion transfomer training",
                group="diffuson transformer",
                project='walker deepRL HW3',
            )

        losses = np.zeros(num_training_steps)
        model.train()
        for training_iter in tqdm(range(num_training_steps)):
            loss = self.training_step(batch_size)
            losses[training_iter] = loss
            if wandb_logging:
                wandb.log({"loss": loss})
            if training_iter % print_every == 0:
                print(f"Training Iteration {training_iter}: loss = {loss}")
            if (training_iter + 1) % save_every == 0:
                # save model in data/diffusion_policy_transformer_models
                os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
                torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy_iter_{training_iter + 1}.pt")

        os.makedirs("data/diffusion_policy_transformer_models", exist_ok=True)
        torch.save(model.state_dict(), f"data/diffusion_policy_transformer_models/diffusion_policy.pt")
        if wandb_logging:
            wandb.finish()
        else:
            x_axis = np.arange(num_training_steps)
            plt.plot(x_axis, losses)
            plt.xlabel("Training Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Diffusion Policy")
            plt.savefig("data/diffusion_policy_transformer_models/diffusion_policy_loss.png")
            print(f"final loss={losses[-1]}")

        return losses
    # TODO: make people write this code
    def training_step(self, batch_size):
        """
        Runs a single training step on the model.

        Args:
            batch_size (int): The batch size to use.

        NOTE: actions_padding is a mask that is False for actions to be predicted and True otherwise 
                (for instance, the model predicts 3 actions, but our batch element may contain the 2 final actions in a sequence)
                when calculating the loss, we should only consider the loss for the actions that are not padded
        NOTE: return a loss value that is a plain float (not a tensor), and is on cpu
        """
        # BEGIN STUDENT SOLUTION

        # state, action, next_action, episode_timesteps, state_padding, action_padding, next_action_padding = self.get_training_batch(batch_size)
        # # state = state.to(self.device)
        # # action = action.to(self.device)
        # # next_action = next_action.to(self.device)
        # # episode_timesteps = episode_timesteps.to(self.device)
        # epsilon = torch.randn(action.shape).to(self.device)
        # t= torch.randint(0, self.num_train_diffusion_timesteps, (next_action.shape[0],1))
        # pred_act = self.training_scheduler.add_noise(action, epsilon, t) #self.training_scheduler.alphas_cumprod[t].to(self.device)
        # #pred_act = torch.sqrt(a_hat) * action + torch.sqrt(1 - a_hat) * epsilon

        (states, 
        prev_actions, 
        actions, 
        episode_timesteps, 
        states_padding, 
        prev_actions_padding, 
        actions_padding) = self.get_training_batch(batch_size, max_action_len=3, num_previous_states=5, num_previous_actions=4)

        # Generate noise that will be added to the actions
        epsilon = torch.randn(actions.shape, device=self.device)

        t = torch.randint(0, self.num_train_diffusion_timesteps, size=(actions.shape[0], 1), device=self.device)

        # alpha_bar = self.training_scheduler.alphas_cumprod[t].squeeze()
        # noised_actions = np.sqrt(alpha_bar) * actions_batch + np.sqrt(1 - alpha_bar) * epsilon
        noisy_actions = self.training_scheduler.add_noise(actions, epsilon, t)

        noise_predictions = self.model(
            states,
            prev_actions,
            noisy_actions,
            episode_timesteps,
            t,
            states_padding,
            prev_actions_padding,
            actions_padding,    
        )

        loss = torch.nn.MSELoss()(noise_predictions, epsilon)

        #loss_fn = nn.MSELoss()
        # pred = self.model(state, action, pred_act, episode_timesteps, t, state_padding, action_padding, next_action_padding)
        #loss = loss_fn(pred, epsilon)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach().cpu().item()

        # END STUDENT SOLUTION

        return loss

    def get_training_batch(self, batch_size, max_action_len=3, num_previous_states=5, num_previous_actions=4):
        """
        get a training batch for the model
        Args:
            batch_size (int): the batch size to use
            max_action_len (int): the maximum number of actions to predict
            num_previous_states (int): the number of previous states to condition on
            num_previous_actions (int): the number of previous actions to condition on
        """
        assert num_previous_states == num_previous_actions + 1, f"num_previous_states={num_previous_states} must be equal to num_previous_actions + 1={num_previous_actions + 1}"

        # get trajectory lengths, so we can sample each trajectory with probability proportional to its length
        # this is equivalent to sampling uniformly from the set of all environment steps
        batch_indices = np.random.choice(
            np.arange(len(self.trajectory_lengths)),
            size=batch_size,
            replace=True,
            p=np.array(self.trajectory_lengths) / sum(self.trajectory_lengths)
        )

        previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch = [], [], [], [], [], [], []
        for i in range(len(batch_indices)):
            # get the start and end index for states to condition on
            end_index_state = np.random.randint(1, self.trajectory_lengths[batch_indices[i]])
            start_index_state = max(0, end_index_state - num_previous_states)

            # get the start and end index for actions to condition on (we predict the action for the final state)
            start_index_previous_actions = start_index_state
            end_index_previous_actions = end_index_state - 1

            # get the start and end index for actions to predict
            start_index_action = end_index_state
            end_index_action = min(self.trajectory_lengths[batch_indices[i]], start_index_action + max_action_len)

            previous_states = self.states[batch_indices[i], start_index_state:end_index_state]
            previous_actions = self.actions[batch_indices[i], start_index_previous_actions:end_index_previous_actions]
            actions = self.actions[batch_indices[i], start_index_action:end_index_action]

            state_dim = previous_states.shape[1]
            action_dim = actions.shape[1]

            state_seq_length = previous_states.shape[0]
            previous_action_seq_length = previous_actions.shape[0]

            # if we have less than the max number of previous states, add some padding (i.e. we're predicting a very early state)
            if state_seq_length < num_previous_states:
                previous_states = np.concatenate([previous_states, np.zeros((num_previous_states - state_seq_length, state_dim))], axis=0)
                previous_actions = np.concatenate([previous_actions, np.zeros((num_previous_actions - previous_action_seq_length, action_dim))], axis=0)
                previous_states_padding_mask = np.concatenate([np.zeros(state_seq_length), np.ones(num_previous_states - state_seq_length)], axis=0)
                previous_actions_padding_mask = np.concatenate([np.zeros(previous_action_seq_length), np.ones(num_previous_actions - previous_action_seq_length)], axis=0)
            else:
                previous_states_padding_mask = np.zeros(num_previous_states)
                previous_actions_padding_mask = np.zeros(num_previous_actions)

            # if we have less than the max number of actions, add some padding (i.e. we're predicting a very early action)
            action_seq_length = actions.shape[0]
            if action_seq_length < max_action_len:
                action_dim = actions.shape[1]
                actions = np.concatenate([actions, np.zeros((max_action_len - action_seq_length, action_dim))], axis=0)
                action_padding_mask = np.concatenate([np.zeros(action_seq_length), np.ones(max_action_len - action_seq_length)], axis=0)

            else:
                action_padding_mask = np.zeros(max_action_len)

            previous_states_batch.append(previous_states)
            previous_actions_batch.append(previous_actions)
            actions_batch.append(actions)
            episode_timesteps_batch.append(np.arange(start_index_state, start_index_state + num_previous_states)) # add extra dummy timesteps in some cases
            previous_states_padding_batch.append(previous_states_padding_mask)
            previous_actions_padding_batch.append(previous_actions_padding_mask)
            actions_padding_batch.append(action_padding_mask)

        previous_states_batch = np.stack(previous_states_batch)
        previous_actions_batch = np.stack(previous_actions_batch)
        actions_batch = np.stack(actions_batch)
        episode_timesteps_batch = np.stack(episode_timesteps_batch)
        previous_states_padding_batch = np.stack(previous_states_padding_batch)
        previous_actions_padding_batch = np.stack(previous_actions_padding_batch)
        actions_padding_batch = np.stack(actions_padding_batch)

        previous_states_batch = torch.from_numpy(previous_states_batch).float().to(self.device)
        previous_actions_batch = torch.from_numpy(previous_actions_batch).float().to(self.device)
        actions_batch = torch.from_numpy(actions_batch).float().to(self.device)
        previous_states_padding_batch = torch.from_numpy(previous_states_padding_batch).bool().to(self.device)
        previous_actions_padding_batch = torch.from_numpy(previous_actions_padding_batch).bool().to(self.device)
        actions_padding_batch = torch.from_numpy(actions_padding_batch).bool().to(self.device)
        episode_timesteps_batch = torch.from_numpy(episode_timesteps_batch).long().to(self.device)

        return previous_states_batch, previous_actions_batch, actions_batch, episode_timesteps_batch, previous_states_padding_batch, previous_actions_padding_batch, actions_padding_batch

def run_training():
    """
    Creates the environment, model, and optimizer, loads the data, and trains/evaluates the model using the TrainDiffusionPolicy class.
    """

    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    # BEGIN STUDENT SOLUTION
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=6,
        state_dim=state_size,
        act_dim=action_size,
        hidden_size=128,
        max_episode_length=1600,
        n_transformer_heads=1,
        device=device
    )

    states_path = f"data/states_diffusion_policy.pkl"
    actions_path = f"data/actions_diffusion_policy.pkl"

    with open(states_path, "rb") as f:
        states = pickle.load(f)
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    T=TrainDiffusionPolicy(env, model, optimizer, states, actions, device=device, num_train_diffusion_timesteps=30)
    #losses=T.train(num_training_steps=50000, batch_size=256, print_every=256, save_every=50000, wandb_logging=True)
    T.evaluation(num_samples=20, num_actions_to_eval_in_a_row=1)
    #print("loss at 10000 iterations: ", losses[10000])
    #print(f"final loss={losses[-1]}")
    #create_gif(T)
    # END STUDENT 
    
def create_gif(Agent: PolicyDiffusionTransformer):

    gif_path = f"data/diffusion.gif"
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