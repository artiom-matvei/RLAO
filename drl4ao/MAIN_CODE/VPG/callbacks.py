from stable_baselines3.common.callbacks import BaseCallback

class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, ep_len=1000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.ep_len = ep_len

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % self.eval_freq == 0:
            # Perform evaluation

            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            for _ in range(self.ep_len):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, *_ = self.eval_env.step(action)
                episode_reward += reward

            mean_reward = episode_reward / self.ep_len
            self.logger.record("eval/mean_reward", mean_reward)
            print(f"Step {self.n_calls}: Mean Reward: {mean_reward}")
        return True