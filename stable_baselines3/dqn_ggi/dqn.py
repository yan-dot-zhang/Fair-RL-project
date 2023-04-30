import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common_ggi.buffers import GGIReplayBuffer
from stable_baselines3.dqn import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn_ggi.policies import GGICnnPolicy, GGIDQNPolicy, GGIMlpPolicy, GGIMultiInputPolicy

SelfDQN = TypeVar("SelfDQN", bound="DQN_GGI")

class DQN_GGI(DQN):
    """
    Deep Q-Network (DQN) GGF version

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param reward_space: Dimension of rewards function
    :param weight_coef: weight for GGF
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "GGIMlpPolicy": GGIMlpPolicy,
        "GGICnnPolicy": GGICnnPolicy,
        "GGIMultiInputPolicy": GGIMultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[GGIDQNPolicy]],
        env: Union[GymEnv, str],
        reward_space: int,
        weight_coef: Union[int, float, np.ndarray],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[GGIReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        
        self.action_dim = env.action_space.n
        self.reward_space = reward_space
        # if weight_coef is scalar, compute weight by 1/weight_coef**i
        if np.isscalar(weight_coef):
            self.weight_coef = np.array([1 / (weight_coef ** i) for i in range(reward_space)])
        elif len(weight_coef) == reward_space:
            self.weight_coef = weight_coef
        else:
            raise TypeError("`weight_coef` should be either scalar or array with length reward_space")

        # call DQN.__init__()
        super().__init__(
            policy = policy,
            env = env,
            learning_rate = learning_rate,
            buffer_size = buffer_size,
            learning_starts = learning_starts,
            batch_size = batch_size,
            tau = tau,
            gamma = gamma,
            train_freq = train_freq,
            gradient_steps = gradient_steps,
            replay_buffer_class = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            optimize_memory_usage = optimize_memory_usage,
            target_update_interval = target_update_interval,
            exploration_fraction = exploration_fraction,
            exploration_initial_eps = exploration_initial_eps,
            exploration_final_eps = exploration_final_eps,
            max_grad_norm = max_grad_norm,
            stats_window_size = stats_window_size,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device = device,
            _init_setup_model = _init_setup_model,
        )


    def _setup_model(self) -> None:
        ##########################
        # rewrite OffPolicyAlgorithm._setup_model()
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                raise NotImplementedError('GGF version ReplayBuffer for spaces.Dict is not implemented')
            else:
                self.replay_buffer_class = GGIReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                reward_space=self.reward_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )
        
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            reward_space = self.reward_space,
            weight_coef = self.weight_coef,
            lr_schedule = self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        ##########################
        # rewrite DQN._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        # For type checker:
        assert isinstance(self.policy, GGIDQNPolicy)
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    # _on_step(self), same of DQN

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target.predict_get_q_values(replay_data.next_observations)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates, shape = (None, action_dim*reward_space)
            current_q_values = self.q_net(replay_data.observations).reshape((-1, self.action_dim, self.reward_space))

            # Retrieve the q-values for the actions from the replay buffer
            replay_action = replay_data.actions.long().reshape((-1,1,1)).expand((-1,-1,self.reward_space))
            current_q_values = th.gather(current_q_values, dim=1, index=replay_action).squeeze()

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    # def predict(), remain the same

    def learn(
            self: SelfDQN, 
            total_timesteps: int, 
            callback: MaybeCallback = None, 
            log_interval: int = 4, 
            tb_log_name: str = "DQN", 
            reset_num_timesteps: bool = True, 
            progress_bar: bool = False
    ) -> SelfDQN:
        return super().learn(
            total_timesteps = total_timesteps, 
            callback = callback, 
            log_interval = log_interval, 
            tb_log_name = tb_log_name, 
            reset_num_timesteps = reset_num_timesteps, 
            progress_bar = progress_bar
        )
    
    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params()
