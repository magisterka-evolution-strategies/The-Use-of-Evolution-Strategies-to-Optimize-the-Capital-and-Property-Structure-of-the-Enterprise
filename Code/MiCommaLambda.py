import math
import random

import numpy as np
import pandas as pd

from Code.Company import Company
from Code.EvolutionPlatform import EvolutionPlatform
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class MiCommaLambda(EvolutionStrategyInterface):
    def __init__(self, evolution_platform: EvolutionPlatform, mi: int, la: int, factor: float):
        super().__init__(evolution_platform)
        self.mi = mi
        self.la = la
        self.factor = factor

    def generate_random_gradient(self):
        values = []
        for _ in range(4):
            values.append(random.uniform(-1, 1))
        fifth_value = -sum(values)
        if -1 <= fifth_value <= 1:
            values.append(fifth_value)
        else:
            return self.generate_random_gradient()
        return values

    def generate_best_company(self, company: Company):
        best_company = None
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
            mutation = np.concatenate([self.generate_random_gradient(), self.generate_random_gradient()])
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

        return best_company

    def generate_offspring(self):
        new_companies = []
        for i, company in enumerate(self.generated_companies):
            child_company = self.generate_best_company(company)
            if child_company is None:
                child_company = company
            new_companies.append(child_company)

        self.generated_companies = new_companies
