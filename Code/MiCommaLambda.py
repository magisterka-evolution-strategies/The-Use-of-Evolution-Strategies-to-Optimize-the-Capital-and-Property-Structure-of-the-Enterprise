import math
import random
import time

import numpy as np
import pandas as pd

from Code.Company import Company
from Code.EvolutionPlatform import EvolutionPlatform
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class MiCommaLambda(EvolutionStrategyInterface):
    def __init__(self, evolution_platform: EvolutionPlatform, name: str, mi: int, la: int, factor: float):
        super().__init__(evolution_platform, name)
        self.mi = mi
        self.la = la
        self.factor = factor

    def generate_random_gradient(self):
        while True:
            values = np.random.uniform(-1, 1, 4)
            fifth_value = -np.sum(values)
            if -1 <= fifth_value <= 1:
                return np.append(values, fifth_value).tolist()

    def generate_best_company(self, company: Company):
        best_company = company
        best_score = -math.inf
        for i in range(self.la):
            parents = random.sample(self.generated_companies, min(self.mi, len(self.generated_companies)))
            parents = list(map(lambda x: x.to_array(), parents))
            df = pd.DataFrame(parents,
                              columns=["NonCurrentAssets", "CurrentAssets",
                                       "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                       "EquityShareholdersOfTheParent", "NonControllingInterests",
                                       "NonCurrentLiabilities",
                                       "CurrentLiabilities",
                                       "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])
            rotation = random.choice([-1, 1])
            mean = df.mean()
            mutation_strength = random.randint(1, 3)
            mutation = np.concatenate([self.generate_random_gradient(), self.generate_random_gradient()])
            mutation *= mutation_strength
            final_change = mean + mutation
            change = rotation * (company.to_dataframe() - final_change) / self.factor
            new_company_values = [x + y for x, y in
                                  zip(company.to_array(), change.values.tolist()[0])]
            if not only_positive_values(new_company_values):
                continue
            predictions = self.structure_change_model.predict(change, verbose=None)
            prediction = predictions[0][0]
            if prediction <= best_score:
                continue
            child_company = Company(*new_company_values)
            if self.outliers_model.predict(child_company.to_dataframe())[0] == -1:
                continue
            best_score = prediction
            best_company = child_company

        return best_company, best_score

    def generate_offspring(self):
        start_time = time.process_time()
        new_companies = []
        for i, company in enumerate(self.generated_companies):
            best_company, best_score = self.generate_best_company(company)
            if best_score != -math.inf:
                if best_score > 0:
                    self.positive_changes_made += 1
                best_company.value = company.value
                best_company.change_company_value(best_score)
            new_companies.append(best_company)

        self.generated_companies = new_companies
        end_time = time.process_time()
        self.evaluation_times.append(end_time - start_time)
