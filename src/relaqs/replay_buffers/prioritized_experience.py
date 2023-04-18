import numpy as np

class PrioritizedExperienceReplay:

    def __init__(self, alpha:float, beta:float) -> None:
        """
        Args:
            alpha (float): Interpolates between uniform sampling and pure prioritized sampling
            beta (float): annealing constant for defining weights that adjust the learning rate
            memory (float): Captures the  temporal difference errors for the model
            
        """
        self.alpha = alpha
        self.beta = beta
        self.memory = None
        self.epsilon = 0.0001

    def store_memory(self,memory):
        self.memory=memory
        
    def sample(self, batchsize):
        """Samples from the memory
        Args:
            batchsize
        Return
            index of sampled remembered observations
        """
        self._create_probability_distribution()
        prob_distribution = [self.memory[index][5] for index, prob in enumerate(self.memory)]
        return np.random.choice(range(len(self.memory)), batchsize, p=prob_distribution).tolist()

    def _create_probability_distribution(self):
        """Create probability distribution used for sampling experiences
        """
        
        # add a small positive constant to avoid zero probabilities
        for index in range(len(self.memory)):
            # Our tuple should be of length 6 (S,A,R,S,Done,Error)
            assert len(self.memory[index]) == 6
            self.memory[index][5] += self.epsilon
            self.memory[index][5] = self.memory[index][5]**self.alpha
         
        normalization_constant = 0
        # Normalize to get a probability distribution
        for index in range(len(self.memory)):
            # Our tuple should be of length 6 (S,A,R,S,Done,Error)
            assert len(self.memory[index]) == 6
            normalization_constant += self.memory[index][5]
        for index in range(len(self.memory)):
            self.memory[index][5] = self.memory[index][5]/normalization_constant

    def create_weights(self):
        """Creates weights for adjusting the training of the model
        """
    
        weights = [1/(len(self.memory[index])*self.memory[index][5]**self.beta) for index in range(len(self.memory))]
        weights = weights/(max(weights))
        weights = weights.tolist()
        return weights
        