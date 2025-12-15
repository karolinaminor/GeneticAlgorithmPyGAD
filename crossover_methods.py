import random


class CrossoverMethods:
    """Class for implementing various crossover methods in real-representation genetic algorithm."""
    
    @staticmethod
    def arithmetic_crossover(gene1, gene2, bound):
        """Perform arithmetic crossover for one gene between two parents."""
        alpha = random.random()
        offspring1 = alpha * float(gene1) + (1 - alpha) * float(gene2)
        offspring2 = alpha * float(gene2) + (1 - alpha) * float(gene1)
        
        if not (bound[0] <= offspring1 <= bound[1]):
            offspring1 = gene1  # Revert to parent gene if out of bounds

        if not (bound[0] <= offspring2 <= bound[1]):
            offspring2 = gene2  # Revert to parent gene if out of bounds

        return offspring1, offspring2
    
    # TO BE CHECKED - NOT SURE HOW TO IMPLEMENT
    @staticmethod
    def linear_crossover(gene1, gene2, bound):
        """Perform linear crossover for one gene between two parents."""
        offspring1 = 0.5 * float(gene1) + 0.5 * float(gene2)
        offspring2 = 1.5 * float(gene1) - 0.5 * float(gene2)
        offspring3 = -0.5 * float(gene1) + 1.5 * float(gene2)
        
        if not (bound[0] <= offspring1 <= bound[1]):
            offspring1 = gene1

        if not (bound[0] <= offspring2 <= bound[1]):
            offspring2 = gene2

        if not (bound[0] <= offspring3 <= bound[1]):
            offspring3 = gene1

        return offspring1, offspring2, offspring3
 
    
    @staticmethod
    def blend_alpha_crossover(gene1, gene2, bound, alpha=0.5):
        """Perform blend alpha crossover for one gene between two parents."""
        x1 = float(gene1)
        x2 = float(gene2)
        d = abs(x1 - x2)
        lower_bound = min(x1, x2) - alpha * d
        upper_bound = max(x1, x2) + alpha * d
        offspring1 = random.uniform(lower_bound, upper_bound)
        offspring2 = random.uniform(lower_bound, upper_bound)

        if not (bound[0] <= offspring1 <= bound[1]):
            offspring1 = gene1
        if not (bound[0] <= offspring2 <= bound[1]):
            offspring2 = gene2

        return offspring1, offspring2
    
    @staticmethod
    def blend_alpha_beta_crossover(gene1, gene2, bound, alpha=0.5, beta=0.5):
        """Perform blend alpha beta crossover for one gene between two parents."""
        x1 = float(gene1)
        x2 = float(gene2)
        d = abs(x1 - x2)
        lower_bound = min(x1, x2) - alpha * d
        upper_bound = max(x1, x2) + beta * d
        offspring1 = random.uniform(lower_bound, upper_bound)
        offspring2 = random.uniform(lower_bound, upper_bound)
        
        if not (bound[0] <= offspring1 <= bound[1]):
            offspring1 = gene1
        if not (bound[0] <= offspring2 <= bound[1]):
            offspring2 = gene2

        return offspring1, offspring2
    
    @staticmethod
    def average_crossover(gene1, gene2, bound):
        """Perform average crossover for one gene between two parents."""
        offspring = (float(gene1) + float(gene2)) / 2

        if not (bound[0] <= offspring <= bound[1]):
            offspring = gene1

        return offspring