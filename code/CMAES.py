import numpy as np
import matplotlib.pyplot as plt
import gym
import functools
import pdb

def cmaes(fn, dim, num_iter=10):
  """Optimizes a given function using CMA-ES.

  Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.

  Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.c
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
  """
  # Hyperparameters
  sigma = 10
  population_size = 100
  p_keep = 0.10  # Fraction of population to keep
  noise = 0.25  # Noise added to covariance to prevent it from going to 0.

  # Initialize the mean and covariance
  mu = np.zeros(dim)
  cov = sigma**2 * np.eye(dim)

  mu_vec = []
  best_sample_vec = []
  mean_sample_vec = []
  n=10
  L= 2
  for t in range(num_iter):
    # WRITE CODE HERE
    print("iteration: ", t)
    samples=np.random.multivariate_normal(mu, cov, n)
    #pdb.set_trace()
    score_list=[]
    for i in range(n):
      score=0
      for j in range(L):
        score += fn(samples[i])
      
      score = score/L
      score_list.append(score)

    #score_list.append(score)

    score= np.array(score_list)
    #pdb.set_trace()
    elite_ind= np.argpartition(score, -L)[-L:]
    best_cols= samples[elite_ind]
    best_sample_vec = [best_cols[:,k] for k in range(best_cols.shape[1])]
    mu=np.mean(best_cols, axis=0)
    #print(best_cols.shape)
    #print(mu.shape)
    #pdb.set_trace()
    cov = np.mean((best_cols- mu[np.newaxis,:])**2, axis=0) + noise*np.eye(dim)
    #pdb.set_trace()
    mu_vec.append(mu)
    mean_sample_vec.append(np.mean(score))
    best_sample_vec.append(np.max(score))
      

    #pass

  return mu_vec, best_sample_vec, mean_sample_vec

def test_fn(x):
  goal = np.array([65, 49])
  return -np.sum((x - goal)**2)

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, params):
  w = params[:4]
  b = params[4]
  #pdb.set_trace()
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

def rl_fn(params, env):
  assert len(params) == 5 , "params should have length 5"
  ## WRITE CODE HERE
  total_rewards=0
  #pdb.set_trace()
  for i in range(1000):
    s,_= env.reset()
    done=False
    trunc=False
    total=0
    while not done and trunc==False :
      #pdb.set_trace()
      a= _get_action(s, params)
      
      s, r, done, trunc , _= env.step(a)
      total += r
    
    #print("total reward: ", total)
    total_rewards += total
  
  total_rewards = total_rewards/1000
  return total_rewards


# mu_vec, best_sample_vec, mean_sample_vec = cmaes(test_fn, dim=2, num_iter=100)

# x = np.stack(np.meshgrid(np.linspace(-10, 100, 30), np.linspace(-10, 100, 30)), axis=-1)
# fn_value = [test_fn(xx) for xx in x.reshape((-1, 2))]
# fn_value = np.array(fn_value).reshape((30, 30))
# plt.figure(figsize=(6, 4))
# plt.contourf(x[:, :, 0], x[:, :, 1], fn_value, levels=10)
# plt.colorbar()
# mu_vec = np.array(mu_vec)
# plt.plot(mu_vec[:, 0], mu_vec[:, 1], 'b-o')
# plt.plot([mu_vec[0, 0]], [mu_vec[0, 1]], 'r+', ms=20, label='initial value')
# plt.plot([mu_vec[-1, 0]], [mu_vec[-1, 1]], 'g+', ms=20, label='final value')
# plt.plot([65], [49], 'kx', ms=20, label='maximum')
# plt.legend()
# plt.show()

env = gym.make('CartPole-v0')
# params= np.array([-1,-1,-1,-1,-1])
# params2= np.array([1,0,1,0,1])
# params3= np.array([0,1,2,3,4])
# rew=rl_fn(params, env)
# print(rew)
# rew2=rl_fn(params2, env)
# print("rew2: ", rew2)
# rew3=rl_fn(params3, env)
# print("rew3: ", rew3)

# fn_with_env = functools.partial(rl_fn, env=env)
# #print("done with env")
# mu_vec, best_sample_vec, mean_sample_vec = cmaes(fn_with_env, dim=5, num_iter=10)