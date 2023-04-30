from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.nn.functional import one_hot

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)

from stable_baselines3.common.type_aliases import Schedule

def compute_GGI(q_values: th.Tensor, weight: th.Tensor):
    sorted_q_values,_ = th.sort(q_values, dim=-1)
    return th.tensordot(sorted_q_values, weight, dims=1)

class GGIQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param reward_space: dimension of rewards
    :param weight_coef: weight for GGF
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_space: int,
        weight_coef: Union[int, float, np.ndarray],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # reward dim
        self.reward_space = reward_space
        # if weight_coef is scalar, compute weight by 1/weight_coef**i
        if np.isscalar(weight_coef):
            self.weight_coef = np.array([1 / (weight_coef ** i) for i in range(reward_space)])
        elif len(weight_coef) == reward_space:
            self.weight_coef = np.array(weight_coef)
        else:
            raise TypeError("`weight_coef` should be either scalar or array with length reward_space")
        # convert to tensor
        self.weight_coef_tensor = th.Tensor(self.weight_coef)

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = self.action_space.n  # number of actions
        self.action_dim = action_dim
        q_net = create_mlp(self.features_dim, action_dim*reward_space, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # For type checker:
        assert isinstance(self.features_extractor, BaseFeaturesExtractor)
        # return shape, (None, action_dim*reward_space)
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # self(observation).shape = (None, action_dim*reward_space)
        # reshape is needed
        q_values = self(observation).reshape((-1, self.action_dim, self.reward_space))
        # compute GGF
        q_ggi = compute_GGI(q_values, self.weight_coef_tensor)
        # Greedy action
        action = q_ggi.argmax(dim=1).reshape(-1)
        return action

    def predict_get_q_values(self, observation: th.Tensor) -> th.Tensor:
        """return next q value for q-learning"""
        q_values = self(observation).reshape((-1, self.action_dim, self.reward_space))
        # compute GGF
        q_ggi = compute_GGI(q_values, self.weight_coef_tensor)
        # Greedy action
        action = q_ggi.argmax(dim=1).reshape((-1,1,1))
        action_expand = action.expand((-1,-1,self.reward_space))
        selected_q_values = th.gather(q_values, dim=1, index=action_expand)
        return selected_q_values.squeeze()

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                reward_space=self.reward_space,
                weight_coef=self.weight_coef
            )
        )
        return data



class GGIDQNPolicy(BasePolicy):
    """
    GGI version Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param reward_space: dimension of rewards
    :param weight_coef: weight for GGF
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_space: int,
        weight_coef: Union[int, float, np.ndarray],
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.reward_space = reward_space
        self.weight_coef = weight_coef

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "reward_space": self.reward_space,
            "weight_coef": self.weight_coef,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net: GGIQNetwork
        self.q_net_target: GGIQNetwork
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> GGIQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return GGIQNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                reward_space=self.reward_space,
                weight_coef=self.weight_coef
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode


GGIMlpPolicy = GGIDQNPolicy


class GGICnnPolicy(GGIDQNPolicy):
    """
    GGF version Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param reward_space: dimension of rewards
    :param weight_coef: weight for GGF
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_space: int,
        weight_coef: Union[int, float, np.ndarray],
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            weight_coef,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class GGIMultiInputPolicy(GGIDQNPolicy):
    """
    GGF version Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param reward_space: dimension of rewards
    :param weight_coef: weight for GGF
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        reward_space: int,
        weight_coef: Union[int, float, np.ndarray],
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            reward_space,
            weight_coef,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
