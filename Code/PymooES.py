import time

from pandas import Series
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from Code.EvolutionPlatform import EvolutionPlatform
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.Pymoo import Pymoo


class PymooES(EvolutionStrategyInterface):
    def __init__(self, evolution_platform: EvolutionPlatform, name: str, mean_changes: Series | float, std_changes: Series | float):
        super().__init__(evolution_platform, name)
        self.mean_changes = mean_changes
        self.std_changes = std_changes

    def generate_offspring(self):
        start_time = time.process_time()
        problem = Pymoo(self.generated_companies, self.mean_changes, self.std_changes, self.structure_change_model, self.outliers_model)

        algorithm = ES(pop_size=len(self.generated_companies), offspring_size=len(self.generated_companies), sigma=0.5)
        termination = get_termination("n_gen", 1)

        minimize(problem,
                 algorithm,
                 termination,
                 verbose=False)

        positive_changes = problem.apply_best_changes()
        self.positive_changes_made += positive_changes
        end_time = time.process_time()
        self.evaluation_times.append(end_time - start_time)

