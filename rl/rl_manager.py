class RLManager:
    """
    Runs episodes of the game and collects training data.
    Works exclusively with real game states — MCTS handles the abstract states internally.

    Supplies u-MCTS with sequences of real game states (phi_k) which it converts
    to abstract states via NNr.
    """

    def __init__(self, gsm, mcts, eb, nnr, nnd, nnp, q=3):
        self.gsm  = gsm   # game state manager — plays the real game
        self.mcts = mcts  # u-MCTS — picks actions
        self.eb   = eb    # episode buffer — stores training data
        self.nnr  = nnr
        self.nnd  = nnd
        self.nnp  = nnp
        self.q    = q     # look-back — how many past states NNr receives

    def run_episode(self):
        """
        Run one episode of the game.
        At each step k:
            1. Build phi_k = last q+1 real game states
            2. Call MCTS to get action, policy, value
            3. Apply action to real game via GSM
            4. Save (state, value, policy, action, reward) to episode
        """
        state         = self.gsm.get_initial_state()
        episode       = []
        done          = False
        state_history = [state]

        while not done:

            # --- build phi_k: last q+1 real states, pad with blank if needed ---
            blank = (0, 0, 0)
            if len(state_history) < self.q + 1:
                padding    = [blank] * (self.q + 1 - len(state_history))
                root_states = padding + state_history
            else:
                root_states = state_history[-(self.q + 1):]

            # --- call MCTS to get action, policy and value ---
            action, policy, value = self.mcts.search(
                root_states, self.nnr, self.nnd, self.nnp
            )
            print(f"action={action}, policy={policy}")

            # --- apply action to the real game ---
            next_state, reward, done = self.gsm.apply_action(action)

            # --- save this step ---
            episode.append((state, value, policy, action, reward))

            # --- update for next step ---
            state = next_state
            state_history.append(state)

        # --- offload episode to buffer ---
        self.eb.save_episode(episode)
        return episode

    def run(self, num_episodes, nnm, training_interval):
        """
        Run num_episodes episodes and train networks every training_interval episodes.
        This is EPISODE_LOOP() from the assignment pseudocode.

        Args:
            num_episodes       : Ne — total number of episodes to run
            nnm                : NeuralNetworkManager — handles training
            training_interval  : It — train every this many episodes
        """
        for episode_num in range(num_episodes):
            print(f"Episode {episode_num + 1} / {num_episodes}")

            episode = self.run_episode()
            total_reward = sum(step[4] for step in episode)
            print(f"  steps={len(episode)}  total_reward={total_reward:.1f}")

            # train every It episodes
            if episode_num % training_interval == 0 and len(self.eb) > 0:
                print(f"  Training networks on {len(self.eb)} episodes...")
                nnm.train_on_buffer(self.eb)
