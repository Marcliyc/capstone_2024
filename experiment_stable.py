from environment import MultiAgentStationaryImplicit
from algorithm import MultiDQN

#from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3 import DQN, A2C
gammas = [0.5]

lrs = [0.0001]
train_freqs = [6]
exploration_fractions = [0.6]
def exponential_schedule(initial_value, decay_rate=4):
    """
    Exponential learning rate schedule.
    :param initial_value: Initial learning rate.
    :param decay_rate: Rate of exponential decay.
    :return: schedule function that takes the current progress (from 1 to 0) and returns the learning rate.
    """
    def schedule(progress_remaining):
        # Exponential decay: initial_value * e^(-decay_rate * (1 - progress_remaining))
        return initial_value * 10**(-decay_rate * (1 - progress_remaining))
    return schedule

for lr in lrs:
  for gamma in gammas:
    for train_freq in train_freqs:
      for exploration_fraction in exploration_fractions:
          # Create the environment
          env = MultiAgentStationaryImplicit(single_done=True)

          # Initialize the Q-learning agent
          model = MultiDQN("MultiInputPolicy", env, verbose=2,batch_size=512,tau=1,learning_rate=exponential_schedule(lr),train_freq=train_freq,gamma=gamma,exploration_fraction=exploration_fraction)#,tensorboard_log='./DQN_Stationary3/')

          # Train the agent
          model.learn(total_timesteps=1000)#,tb_log_name=f'lr{lr}_discountfactor{gamma}_trainfreq{train_freq}_explorationfraction{exploration_fraction}_unique')
          model.save("dqn_model_unique")