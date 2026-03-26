import math
import random
import torch

class MCTSNode:
    def __init__(self, state, parent=None, action=None, reward=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.reward = reward
        self.visit_count = 0
        self.q_value = 0.0
        self.children = []

    def is_leaf(self):
        """A leaf node has no childeren yet"""
        return len(self.children) == 0
    
    def expand(self, nnd, nnp):
        """
        Add a child per legal action from this node.
        """
        node = self

        #print(f"NODE STATE{node.state}")
        policy, _ = nnp.predict(node.state)

        for action in range(len(policy)):
            next_abstract_state, reward = nnd.predict(node.state, action)
            
            child = MCTSNode(
                state   = next_abstract_state,
                parent  = node,
                action  = action # the action that lead it here
            )
            node.children.append(child)

    def ucb1(self, parent):
        """Calculate the UCB1 score for this node"""
        node = self
        C = 1.4  # explorasion rate
        
        if node.visit_count == 0:
            return float('inf')  # prioritize unvisited nodes
        return node.q_value + C * math.sqrt(math.log(parent.visit_count) / node.visit_count)

    def select(self):
        """
        Travel downwards the tree and pick the best child all the way
        until we find a leaf node. Return the leaf node.
        """
        node = self
        while not node.is_leaf():
            node = max(node.children, key=lambda child: child.ucb1(node)) # find the child with highest UCB1 score
        return node

    def rollout(self, depth, nnd, nnp):
        """
        Simulate the game by the number of times in depth from this node and summarize rewards.
        Return total accumulated rewards.
        """
        state = self.state

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
        accum_reward = []

        for d in range(depth): 
            policy, _ = nnp.predict(state)
            action = random.choices(range(len(policy)), weights=policy, k=1)[0]
            state, reward = nnd.predict(state, action)
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            accum_reward.append(reward)

        # value of the end state
        _, value = nnp.predict(state)
        accum_reward.append(value)
        return accum_reward
    
    def backprop(self, rewards):
        """
        We'll have to send the value from rollout back to the root.
        Update visit_count and q-value for each node on the path.
        """
        node = self
        node.visit_count += 1
        node.q_value += (sum(rewards) - node.q_value) / node.visit_count

        if node.parent is not None:
            rewards.append(node.reward) 
            node.parent.backprop(rewards)
        
class MCTS:
    def __init__(self, num_searches=50, rollout_depth=5):
        self.num_searches = num_searches
        self.rollout_depth = rollout_depth

    def search(self, root_states, nnr, nnd, nnp):
        """
        Run MCTS from a given state and return
        the best action + policy + value
        """
        node = self
        root_tensor = torch.tensor(root_states, dtype=torch.float32)  # [history_len, state_dim]
        abstract_state = nnr.represent(root_tensor)
        root = MCTSNode(abstract_state)
        root.expand(nnd, nnp)

        for step in range(node.num_searches):
            # 1. SELECT - visit a leaf
            leaf = root.select()

            # 2. EXPAND - expand child if it hasen't been visited
            if leaf.visit_count > 0:
                leaf.expand(nnd, nnp)
                if leaf.children:
                    leaf = random.choice(leaf.children)

            # 3. ROLLOUT - estimate value
            accum_reward = leaf.rollout(node.rollout_depth, nnd, nnp)

            # 4. BACKPROP - send value back up
            leaf.backprop(accum_reward)

        # pick best action based on number of visitis after all searches
        best_child = max(root.children, key=lambda c: c.visit_count)
        
        # policy
        total_visits = sum(child.visit_count for child in root.children)
        policy = [0.0,0.0,0.0]
        for child in root.children:
            policy[child.action] = child.visit_count / total_visits # how often did mcts choose each action
        
        # value of the root node
        value = root.q_value

        return best_child.action, policy, value
