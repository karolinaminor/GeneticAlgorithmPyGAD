import random


class MutationMethods:
    """Class implementing mutation operations for real-representation chromosomes."""
    
    @staticmethod
    def uniform_mutation(genes, p_mutation, bounds):
        """
        Uniform mutation: each gene has p probability of mutation.
        """
        new_genes = []
        for gene, bound in zip(genes, bounds):
            if random.random() < p_mutation:
                mutated_gene = random.uniform(bound[0], bound[1])
                new_genes.append(mutated_gene)
            else:
                new_genes.append(gene)
        return new_genes
    
    @staticmethod
    def gaussian_mutation(genes, p_mutation, bounds, sigma=0.1):
        """
        Gaussian mutation: each gene has p probability of mutation by adding Gaussian noise.
        """
        new_genes = []
        for gene, bound in zip(genes, bounds):
            if random.random() < p_mutation:
                noise = random.gauss(0, sigma)
                mutated_gene = gene + noise
                # Ensure mutated gene is within bounds
                if not (bound[0] <= mutated_gene <= bound[1]):
                    mutated_gene = gene  # Revert to original if out of bounds
                new_genes.append(mutated_gene)
            else:
                new_genes.append(gene)
        return new_genes