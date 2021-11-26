import gym
from .policies import QPolicy

class Q:
    def __init__(self, 
        env: gym.Env,  
        policy: QPolicy, 
        gamma: float, 
        alpha: float, 
    ):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha


    def run_episode(self, 
        max_steps: int = None, 
        render: bool = False, 
        render_mode: str = None,
    ):

        # Reset the environment and get starting state.
        curr_state = self.env.reset()

        total_reward = 0
        step = 0
        while True:

            # Render the environment if requested.
            if render: self.env.render(mode=render_mode)

            # Step the algorithm through the current state and retreive
            # the Q-matrix, next state, and the termination flag.
            next_state, reward, done = self.policy.step(
                curr_state, 
                self.env, 
                self.gamma, 
                self.alpha,
            )

            # Accumulate rewards for the current episode.
            total_reward += reward

            # Update the current state.
            curr_state = next_state

            # Update the step count.
            step += 1

            # Terminate steps early if environment enters terminal state.
            if done: break

            # Terminate if step count is reached.
            elif max_steps is not None and step >= max_steps: break

        return total_reward

