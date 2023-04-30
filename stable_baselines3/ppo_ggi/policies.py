# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common_ggi.policies import GGIActorCriticPolicy, GGIActorCriticCnnPolicy, GGIMultiInputActorCriticPolicy

GGIMlpPolicy = GGIActorCriticPolicy
GGICnnPolicy = GGIActorCriticCnnPolicy
GGIMultiInputPolicy = GGIMultiInputActorCriticPolicy
