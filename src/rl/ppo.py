from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv

from luxai_s3.wrappers import LuxAIS3GymEnv
from rl.wrappers import RLWrapper
from rl.extractor import CustomBoardFeatureExtractor

env = RLWrapper(LuxAIS3GymEnv(numpy_output=True))
check_env(env)

# TODO: match_count_per_episode is set to 1 for RL training
# TODO: player is set to random for RL training
env = DummyVecEnv([lambda: Monitor(RLWrapper(LuxAIS3GymEnv(numpy_output=True))) for _ in range(4)])
env = VecNormalize(env, norm_reward=True, norm_obs=False)

# Stop training when the model reaches the reward threshold
callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10_000, verbose=1)
eval_callback = EvalCallback(env,
                             best_model_save_path="logs/best",
                             callback_after_eval=callback_on_best,
                             verbose=1)

log_path = "logs/ppo_logs/v0"
tensorboard = configure(log_path, ["tensorboard"])
# tensorboard = configure(log_path, ["stdout", "tensorboard"])

# n_step = num steps * num game in episode
# batch_size = n_env*n_step / 2
policy_kwargs = dict(
    features_extractor_class=CustomBoardFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, n_steps=505, batch_size=1010, learning_rate=1e-4,
            verbose=1, clip_range_vf=0.2, ent_coef=0.1)
# model = PPO.load("./logs/best/best_model.zip", env=env)

model.set_logger(tensorboard)

model.learn(total_timesteps=5_000_000, callback=eval_callback, reset_num_timesteps=False)
model.save('logs/models/baseline.bin')
