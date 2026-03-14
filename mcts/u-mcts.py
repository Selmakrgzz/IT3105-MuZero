import math

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
