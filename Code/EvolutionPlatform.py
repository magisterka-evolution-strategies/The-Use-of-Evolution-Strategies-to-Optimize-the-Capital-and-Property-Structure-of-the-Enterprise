import csv
import os
from typing import List

from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Company import Company
from Code.utils.generate_company_structure import generate_structure_mean


class EvolutionPlatform:
    def __init__(self, outliers_model, structure_change_model):
        self.outliers_model = outliers_model
        self.structure_change_model = structure_change_model
        self.generated_companies = []
        self.evolution_strategies: List[EvolutionStrategyInterface] = []

        self.data_path = "data/start_companies.csv"

    def generate_start_companies(self, number_of_companies, means):
        start_len = len(self.generated_companies)

        while len(self.generated_companies) < number_of_companies:
            assets = generate_structure_mean(means[:5], 5)
            liabilities = generate_structure_mean(means[5:], 5)
            company = Company(*assets, *liabilities)
            if self.outliers_model.predict(company.to_dataframe())[0] == -1:
                continue
            self.generated_companies.append(company)
            print("Company structure generated. Total structures:", len(self.generated_companies))

        if start_len >= number_of_companies:
            return
        with open(self.data_path, "a", newline="") as file:
            tmp = list(map(lambda x: x.to_array(), self.generated_companies[start_len:]))
            writer = csv.writer(file)
            writer.writerows(tmp)

    def load_companies(self, number_of_companies):
        if not os.path.exists(self.data_path):
            return
        with open(self.data_path, "r") as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            tmp = [row for row in reader]
            tmp = tmp[:min(number_of_companies, len(tmp))]
            for structure in tmp:
                company = Company(*structure)
                self.generated_companies.append(company)

    def check_generated_structures(self):
        for company in self.generated_companies:
            values = company.to_dataframe().values[0]
            print(["{:.2f}".format(num) for num in values])

    def add_evolution_strategy(self, evolution_strategy: EvolutionStrategyInterface):
        self.evolution_strategies.append(evolution_strategy)

    def start_evolution(self, epochs: int):
        for epoch in range(epochs):
            print(epoch)
            for evolution_strategy in self.evolution_strategies:
                evolution_strategy.generate_offspring()
                evolution_strategy.check_generated_structures()