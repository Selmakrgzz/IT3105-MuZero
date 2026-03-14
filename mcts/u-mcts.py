import math
import random

C = 1.4  # explorasion rate

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.visits   = 0
        self.q_value  = 0.0
        self.children = []

    def is_leaf(self):
        """A leaf node has no childeren yet"""
        return len(self.children) == 0
    
    def expand(self, gsm):
        """
        Add a child per legal action from this node.
        We'll use gsm to find the legal actions and next state.
        """
        legal_actions = gsm.get_legal_actions(self.state)

        for action in legal_actions:
            next_state, reward = gsm.get_next_state_and_reward(self.state, action)
            child = MCTSNode(
                state   = next_state,
                parent  = self,
                action  = action # the action that lead it here
            )
            self.children.append(child)

    def ucb1(self, parent):
        """Calculate the UCB1 score for this node"""
        if self.visits == 0:
            return float('inf')  # prioritize unvisited nodes
        return self.q_value + C * math.sqrt(math.log(parent.visits) / self.visits)

    def select(self):
        """
        Travel downwards the tree and pick the best child all the way
        until we find a leaf node. Return the leaf node.
        """
        node = self
        while not node.is_leaf():
            node = max(node.children, key=lambda child: child.ucb1(node)) # find the child with highest UCB1 score
        return node

    def rollout(self, gsm, depth=5):
        """
        Simulate the game by the number of times in depth from this node and summarize rewards.
        Return total accumulated rewards.
        """
        state  = self.state
        total  = 0.0

        for step in range(depth):
            actions = gsm.get_legal_actions(state)
            action  = random.choice(actions) # random action for now
            state, reward = gsm.get_next_state_and_reward(state, action)
            total += reward

        # add herustic value of the end state
        total += gsm.evaluate_state(state)
        return total
    
    def backprop(self, value):
        """
        We'll have to send the value from rollout back to the root.
        Update visits and q-value for each node on the path.
        """
        node = self
        while node is not None:
            node.visits  += 1
            node.q_value += (value - node.q_value) / node.visits
            node = node.parent
    
