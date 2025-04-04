from Code import EvolutionPlatform


class EvolutionStrategyInterface:
    def __init__(self, evolution_platform: EvolutionPlatform):
        self.generated_companies = evolution_platform.generated_companies.copy()
        self.outliers_model = evolution_platform.outliers_model
        self.structure_change_model = evolution_platform.structure_change_model

    def check_generated_structures(self):
        for company in self.generated_companies:
            values = company.to_dataframe().values[0]
            print(["{:.2f}".format(num) for num in values])

    def generate_offspring(self):
        pass