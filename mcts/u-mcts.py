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

