
class JJ :
    def __init__(self, agents, alpha, beta, rank=0) :
        self.agents = agents
        self.alpha = alpha
        self.beta = beta
        self.rank_history = [rank]
        self.smurf_values = []
        self.games = []

    def rank(self) :
        return self.rank_history[-1]

    def update_rank(self, gain) :
        self.rank_history.append(self.rank() + gain)

    def set_rank(self, rank) :
        self.rank_history.append(rank)

hidden_layers_sizes = [512, 256, 128, 64, 32]

game = "connect_four"

agent_path = "agents/better_puissance4/"

jjs_path = 'data/better_puissance4_fast_2.pkl'
