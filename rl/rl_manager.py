# rl_manager.py

from collections import deque
from typing import Any, Callable, Optional

import torch

from state_managers.gsm import GameStateManager
from env.vgs import GRID_WIDTH, GRID_HEIGHT, BASKET_WIDTH
from neural_networks.nn_manager import NeuralNetworkManager

class ReinforcementLearningManager:
    """
    High-level MuZero episode runner.

    Responsibilities:
    - reset the real game
    - build state histories phi_k = {s_{k-q}, ..., s_k}
    - ask u-MCTS (or NN fallback) for policy/value at the root
    - sample an action from that policy
    - step the real game
    - store episode data for the episode buffer
    """

    def __init__(
        self,
        gsm: GameStateManager,
        nn_manager: NeuralNetworkManager,
        episode_buffer: Optional[Any] = None,
        umcts: Optional[Any] = None,
        q: int = 0,
        max_steps: int = 200,
        device: Optional[str] = None,
        temperature: float = 1.0,
        state_encoder: Optional[Callable[[tuple], torch.Tensor]] = None,
    ):
        self.gsm = gsm
        self.nn_manager = nn_manager
        self.episode_buffer = episode_buffer
        self.umcts = umcts

        self.q = q
        self.history_len = q + 1
        self.max_steps = max_steps
        self.temperature = max(temperature, 1e-8)

        self.device = device or getattr(nn_manager, "device", "cpu")
        self.state_encoder = state_encoder or self._default_state_encoder

        self.recent_states = deque(maxlen=self.history_len)
        self.current_state = None
        self.current_episode = None

    # -------------------------
    # state helpers
    # -------------------------

    def _default_state_encoder(self, state: tuple) -> torch.Tensor:
        """
        Encode catcher game state (ball_x, ball_y, basket_x) as normalized floats.
        """
        ball_x, ball_y, basket_x = state
        return torch.tensor(
            [
                ball_x / max(1, GRID_WIDTH - 1),
                ball_y / max(1, GRID_HEIGHT - 1),
                basket_x / max(1, GRID_WIDTH - BASKET_WIDTH),
            ],
            dtype=torch.float32,
        )

    def _blank_state(self) -> tuple:
        """
        Blank state used when k < q.
        """
        return tuple(0 for _ in range(self.gsm.get_state_size()))

    def _new_episode_record(self):
        return []

    def reset(self):
        """
        Start a fresh real game episode.
        """
        if hasattr(self.gsm, "clear_cache"):
            self.gsm.clear_cache()

        self.current_state = self.gsm.get_initial_state()
        self.recent_states.clear()
        self.recent_states.append(self.current_state)
        self.current_episode = self._new_episode_record()

        return self.current_state

    def _build_state_stack(self) -> torch.Tensor:
        """
        Build phi_k = {s_{k-q}, ..., s_k}, padded with blank states at the front.

        Returns tensor of shape [1, history_len, state_dim]
        """
        states = list(self.recent_states)

        while len(states) < self.history_len:
            states.insert(0, self._blank_state())

        encoded = [self.state_encoder(s) for s in states]
        state_stack = torch.stack(encoded, dim=0).unsqueeze(0)  # [1, H, state_dim]
        return state_stack.to(self.device)

    # -------------------------
    # policy/value helpers
    # -------------------------

    @torch.no_grad()
    def _predict_root_with_nn(self, state_stack: torch.Tensor):
        """
        Get root latent, policy, and value from the neural network manager.
        """
        if hasattr(self.nn_manager, "predict_root"):
            root_latent, policy_probs, value = self.nn_manager.predict_root(state_stack)
        else:
            root_latent, policy_logits, value = self.nn_manager.initial_inference(state_stack)
            policy_probs = torch.softmax(policy_logits, dim=-1)

        policy_probs = policy_probs.squeeze(0).detach().cpu()  # [action_size]
        value = float(torch.as_tensor(value).reshape(-1)[0].item())

        return root_latent, policy_probs, value

    def _mask_and_normalize_policy(self, policy, legal_actions) -> torch.Tensor:
        """
        Zero out illegal actions and renormalize.
        Accepts either:
        - full policy over all actions
        - policy only over legal actions
        """
        action_size = self.gsm.get_action_size()
        policy = torch.as_tensor(policy, dtype=torch.float32).flatten()

        masked = torch.zeros(action_size, dtype=torch.float32)

        if policy.numel() == action_size:
            masked[:] = policy[:action_size]
        elif policy.numel() == len(legal_actions):
            for i, action in enumerate(legal_actions):
                masked[action] = policy[i]
        else:
            raise ValueError(
                f"Policy has wrong size. Got {policy.numel()}, expected "
                f"{action_size} or {len(legal_actions)}."
            )

        for a in range(action_size):
            if a not in legal_actions:
                masked[a] = 0.0

        total = masked.sum().item()
        if total <= 0:
            masked[legal_actions] = 1.0 / len(legal_actions)
        else:
            masked = masked / total

        return masked

    def _get_policy_and_value(self, state, state_stack: torch.Tensor):
        """
        Query u-MCTS if available. Otherwise use the NN directly.
        Returns:
            policy: torch.Tensor [action_size]
            value: float
        """
        legal_actions = self.gsm.get_legal_actions(state)

        root_latent, nn_policy, nn_value = self._predict_root_with_nn(state_stack)

        # Fallback: no MCTS yet, just use the NN policy/value directly.
        if self.umcts is None:
            return self._mask_and_normalize_policy(nn_policy, legal_actions), nn_value

        # Expected MCTS interface:
        # search(...) -> (policy, value) OR {"policy": ..., "value": ...}
        #
        # Adapt this call if your MCTS method uses different argument names.
        nnd,nnr,nnp = self.nn_manager.get_nn()
        search_out = self.umcts.search(
            root_game_state=state,
            root_latent=root_latent,
            legal_actions=legal_actions,
            gsm=self.gsm,
            nn_manager=self.nn_manager,
        )

        if isinstance(search_out, dict):
            policy = search_out["policy"]
            value = search_out["value"]
        else:
            policy, value = search_out

        policy = self._mask_and_normalize_policy(policy, legal_actions)
        value = float(torch.as_tensor(value).reshape(-1)[0].item())

        return policy, value

    def _select_action(self, policy: torch.Tensor, deterministic: bool = False) -> int:
        """
        Sample from the policy distribution, or take argmax for evaluation.
        """
        policy = torch.as_tensor(policy, dtype=torch.float32)

        if deterministic:
            return int(torch.argmax(policy).item())

        if self.temperature != 1.0:
            policy = policy.pow(1.0 / self.temperature)
            policy = policy / policy.sum()

        action = torch.multinomial(policy, num_samples=1).item()
        return int(action)

    # -------------------------
    # episode stepping
    # -------------------------

    def step(self, deterministic: bool = False):
        """
        Do one real environment step:
        - build state history
        - get root policy/value
        - choose action
        - apply action in the real game
        - record transition
        """
        if self.current_state is None:
            raise RuntimeError("Call reset() before step().")

        state_before = self.current_state
        state_stack = self._build_state_stack()

        policy, value = self._get_policy_and_value(state_before, state_stack)
        action = self._select_action(policy, deterministic=deterministic)

        next_state, reward, done = self.gsm.apply_action(action)

        # Store exactly what MuZero episode data needs at this timestep.
        self.current_episode["states"].append(state_before)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(float(reward))
        self.current_episode["policies"].append(policy.tolist())
        self.current_episode["values"].append(float(value))

        self.current_state = next_state
        self.recent_states.append(next_state)

        self.current_episode.append({
            "state": state_before,
            "value": float(value),
            "policy": policy.tolist(),
            "action": int(action),
            "reward": float(reward),
        })

    def run_episode(self, deterministic: bool = False, store: bool = True) -> dict:
        """
        Run one full episode in the real game.
        """
        self.reset()

        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < self.max_steps:
            info = self.step(deterministic=deterministic)
            total_reward += info["reward"]
            done = info["done"]
            steps += 1

        episode = self.snapshot_episode()

        if store:
            self.offload_episode(episode)

        return {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
        }

    def run_episodes(
        self,
        num_episodes: int,
        deterministic: bool = False,
        train_interval: Optional[int] = None,
        train_fn: Optional[Callable[[], None]] = None,
    ) -> list[dict]:
        """
        Run multiple episodes.
        Optionally call train_fn every train_interval episodes.
        """
        results = []

        for ep in range(num_episodes):
            summary = self.run_episode(deterministic=deterministic, store=True)
            results.append(summary)

            if (
                train_interval is not None
                and train_fn is not None
                and (ep + 1) % train_interval == 0
            ):
                train_fn()

        return results

    # -------------------------
    # episode buffer helpers
    # -------------------------

    def snapshot_episode(self) -> dict:
        """
        Return a copy of the current episode record.
        """
        return list(self.current_episode)

    def offload_episode(self, episode: Optional[dict] = None):
        """
        Save the episode into the episode buffer.
        """
        if episode is None:
            episode = self.snapshot_episode()

        if self.episode_buffer is None:
            return episode

        if hasattr(self.episode_buffer, "add_episode"):
            self.episode_buffer.add_episode(episode)
        elif hasattr(self.episode_buffer, "store_episode"):
            self.episode_buffer.store_episode(episode)
        elif hasattr(self.episode_buffer, "append"):
            self.episode_buffer.append(episode)
        else:
            raise AttributeError(
                "Episode buffer must have add_episode(...), store_episode(...), or append(...)."
            )

        return episode