import system
from system import MazeSwitches
from system.Wrappers.OneHotEncodingWrapper import OHEWrapper
from system.Wrappers.StateScaleWrapper import StateScaleWrapper
from model.policy.stochastic.CategoricalDistModel import CategoricalDistModel
from agent.trainable.pure.StochasticAgent import StochasticAgent
from runner.BatchSampling import BatchSampling


def initialize_experiment(exp_nb=1):
    if exp_nb == 1 or exp_nb == 2:
        env = MazeSwitches(action_cost=1.).initialize(horizon=100, device="cpu", gamma=0.98)
    elif exp_nb == 3:
        env = MazeSwitches(action_cost=0.).initialize(horizon=40, device="cpu", gamma=0.95)
    else:
        raise NotImplementedError

    env_wrapped = OHEWrapper().initialize(env)
    # env_wrapped = StateScaleWrapper().initialize(env, scale=[4., 12., 1.])
    model = CategoricalDistModel().initialize(env_wrapped, layers=(64, 64, 64,))
    agent = StochasticAgent(model).initialize(env_wrapped)
    sampler = BatchSampling(env_wrapped, agent)

    return env, env_wrapped, model, agent, sampler
