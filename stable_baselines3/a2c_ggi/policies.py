# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common_ggi.policies import GGIActorCriticCnnPolicy, GGIActorCriticPolicy, GGIMultiInputActorCriticPolicy

GGIMlpPolicy = GGIActorCriticPolicy
GGICnnPolicy = GGIActorCriticCnnPolicy
GGIMultiInputPolicy = GGIMultiInputActorCriticPolicy
