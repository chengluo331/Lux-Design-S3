from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode, SingleAgentWrapper
from luxai_s3.params import EnvParams
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from kits.python.agent import Agent
from stable_baselines3.common.logger import configure

env = SingleAgentWrapper(LuxAIS3GymEnv(numpy_output=True), 'player_0', Agent)

check_env(env)

log_path = "./ppo_logs"
tensorboard = configure(log_path, ["stdout", "tensorboard"])

model = PPO("MultiInputPolicy", env, verbose=1, n_steps=100)
model.set_logger(tensorboard)

model.learn(total_timesteps=10_000)