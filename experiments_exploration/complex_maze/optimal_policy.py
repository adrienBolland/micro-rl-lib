import torch
from experiments_exploration.complex_maze.initialize_problem import initialize_experiment


def optimal_policy(exp_nb):
    env, _, model, agent, sampler = initialize_experiment(exp_nb)

    env.reset(1)
    action_sequence = [2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 1, 1]
    cum_r = 0.
    done = False
    gamma = 1.
    for a in action_sequence:
        _, reward, done, _ = env.step(torch.tensor([[a]]))
        cum_r += (gamma * reward.item())
        gamma *= env.gamma

    while not done:
        _, reward, done, _ = env.step(torch.tensor([[0.]]))
        cum_r += (gamma * reward.item())
        gamma *= env.gamma

    print(cum_r)


if __name__ == "__main__":
    optimal_policy(1)  # 2726
    optimal_policy(2)  # 2726
    optimal_policy(3)  # 497
