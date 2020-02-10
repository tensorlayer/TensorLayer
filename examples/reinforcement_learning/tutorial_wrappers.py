"""Env wrappers
Note that this file is adapted from `https://pypi.org/project/gym-vec-env` and
`https://github.com/openai/baselines/blob/master/baselines/common/*wrappers.py`
"""
from collections import deque
from functools import partial
from multiprocessing import Pipe, Process, cpu_count
from sys import platform

import cv2
import gym
import numpy as np
from gym import spaces

__all__ = (
    'build_env',  # build env
    'TimeLimit',  # Time limit wrapper
    'NoopResetEnv',  # Run random number of no-ops on reset
    'FireResetEnv',  # Reset wrapper for envs with fire action
    'EpisodicLifeEnv',  # end-of-life == end-of-episode wrapper
    'MaxAndSkipEnv',  # skip frame wrapper
    'ClipRewardEnv',  # clip reward wrapper
    'WarpFrame',  # warp observation wrapper
    'FrameStack',  # stack frame wrapper
    'LazyFrames',  # lazy store wrapper
    'RewardScaler',  # reward scale
    'SubprocVecEnv',  # vectorized env wrapper
    'VecFrameStack',  # stack frames in vectorized env
    'Monitor',  # Episode reward and length monitor
)
cv2.ocl.setUseOpenCL(False)
# env_id -> env_type
id2type = dict()
for _env in gym.envs.registry.all():
    id2type[_env.id] = _env._entry_point.split(':')[0].rsplit('.', 1)[1]


def build_env(env_id, vectorized=False, seed=0, reward_scale=1.0, nenv=0):
    """Build env based on options"""
    env_type = id2type[env_id]
    nenv = nenv or cpu_count() // (1 + (platform == 'darwin'))
    stack = env_type == 'atari'
    if not vectorized:
        env = _make_env(env_id, env_type, seed, reward_scale, stack)
    else:
        env = _make_vec_env(env_id, env_type, nenv, seed, reward_scale, stack)

    return env


def _make_env(env_id, env_type, seed, reward_scale, frame_stack=True):
    """Make single env"""
    if env_type == 'atari':
        env = gym.make(env_id)
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env)
        # deepmind wrap
        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        if frame_stack:
            env = FrameStack(env, 4)
    elif env_type == 'classic_control':
        env = Monitor(gym.make(env_id))
    else:
        raise NotImplementedError
    if reward_scale != 1:
        env = RewardScaler(env, reward_scale)
    env.seed(seed)
    return env


def _make_vec_env(env_id, env_type, nenv, seed, reward_scale, frame_stack=True):
    """Make vectorized env"""
    env = SubprocVecEnv([partial(_make_env, env_id, env_type, seed + i, reward_scale, False) for i in range(nenv)])
    if frame_stack:
        env = VecFrameStack(env, 4)
    return env


class TimeLimit(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):

    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):

    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames so it's important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        shape = (2, ) + env.observation_space.shape
        self._obs_buffer = np.zeros(shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = info = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):

    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        shape = (self.height, self.width, 1 if self.grayscale else 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        size = (self.width, self.height)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also `LazyFrames`
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k, )
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.asarray(self._get_ob()), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):

    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can be
        huge for DQN's 1M frames replay buffers.

        This object should only be converted to numpy array before being passed
        to the model. You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class RewardScaler(gym.RewardWrapper):
    """Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance drastically.
    """

    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class VecFrameStack(object):

    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.action_space = env.action_space
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k, )
        self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.asarray(self._get_ob())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.asarray(self._get_ob()), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env._reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.num_envs = len(env_fns)

        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        zipped_args = zip(self.work_remotes, self.remotes, env_fns)
        self.ps = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zipped_args
        ]

        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def _step_async(self, actions):
        """
            Tell all the environments to start taking a step
            with the given actions.
            Call step_wait() to get the results of the step.
            You should not call this if a step_async run is
            already pending.
            """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        """
            Wait for the step taken with step_async().
            Returns (obs, rews, dones, infos):
             - obs: an array of observations, or a tuple of
                    arrays of observations.
             - rews: an array of rewards
             - dones: an array of "episode done" booleans
             - infos: a sequence of info objects
            """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
            Reset all the environments and return an array of
            observations, or a tuple of observation arrays.
            If step_async is still doing work, that work will
            be cancelled and step_wait() should not be called
            until step_async() is invoked again.
            """
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def _reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()


class Monitor(gym.Wrapper):

    def __init__(self, env):
        super(Monitor, self).__init__(env)
        self._monitor_rewards = None

    def reset(self, **kwargs):
        self._monitor_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        o_, r, done, info = self.env.step(action)
        self._monitor_rewards.append(r)
        if done:
            info['episode'] = {'r': sum(self._monitor_rewards), 'l': len(self._monitor_rewards)}
        return o_, r, done, info


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


def unit_test():
    env_id = 'CartPole-v0'
    unwrapped_env = gym.make(env_id)
    wrapped_env = build_env(env_id, False)
    o = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, o.shape))
    done = False
    while not done:
        a = unwrapped_env.action_space.sample()
        o_, r, done, info = wrapped_env.step(a)
        print('Take action {} get reward {} info {}'.format(a, r, info))

    env_id = 'PongNoFrameskip-v4'
    nenv = 2
    unwrapped_env = gym.make(env_id)
    wrapped_env = build_env(env_id, True, nenv=nenv)
    o = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, o.shape))
    for _ in range(1000):
        a = [unwrapped_env.action_space.sample() for _ in range(nenv)]
        a = np.asarray(a, 'int64')
        o_, r, done, info = wrapped_env.step(a)
        print('Take action {} get reward {} info {}'.format(a, r, info))


if __name__ == '__main__':
    unit_test()
