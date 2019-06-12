class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """

    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)

        # Discount rate
        self.gamma = gamma

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam

        # Total timesteps taken
        self.total_timesteps = total_timesteps

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones = [], [], [], [], []

        # For n in range number of steps
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because AbstractEnvRunner run self.obs[:] = env.reset()
            actions, values = self.model.step(self.obs, self.dones)

            # print("actions runner runner", actions)

            # Append the observations into the mb
            mb_obs.append(np.copy(self.obs))  # obs len nenvs (1 step per env)

            # Append the actions taken into the mb
            mb_actions.append(actions)

            # Append the values calculated into the mb
            mb_values.append(values)

            # Append the dones situations into the mb
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)

            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)

        ### GENERALIZED ADVANTAGE ESTIMATION
        # discount/bootstrap off value fn
        # We create mb_returns and mb_advantages
        # mb_returns will contain Advantage + value
        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)

        lastgaelam = 0

        # From last step to first step
        for t in reversed(range(self.nsteps)):
            # If t == before last step
            if t == self.nsteps - 1:
                # If a state is done, nextnonterminal = 0
                # In fact nextnonterminal allows us to do that logic

                # if done (so nextnonterminal = 0):
                #    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0)
                # else (not done)
                # delta = R + gamma * V(st+1)
                nextnonterminal = 1.0 - self.dones

                # V(t+1)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]

                nextvalues = mb_values[t + 1]

            # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
            mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        # Returns
        mb_returns = mb_advantages + mb_values

        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values))


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])