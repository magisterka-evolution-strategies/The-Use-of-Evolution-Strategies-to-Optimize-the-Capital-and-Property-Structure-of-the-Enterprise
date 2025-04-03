import random

import numpy as np
import pandas as pd

from Code.Company import Company
from Code.EvolutionPlatform import EvolutionPlatform
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class MiPlusLambda(EvolutionStrategyInterface):
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
        best_company = company
        best_score = 0
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
            child_company.change_company_value(prediction)
            best_score = prediction
            best_company = child_company

        return best_company

    def generate_offspring(self):
        new_companies = self.generated_companies.copy()

        i = 0
        while i < self.la:
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
            # parent = random.choice(self.generated_companies)
            parent = random.randint(0, len(self.generated_companies) - 1)
            change = rotation * (new_companies[parent].to_dataframe() - mean) / self.factor
            mutation = np.concatenate([self.generate_random_gradient(), self.generate_random_gradient()])
            # final_change = change
            final_change = change + mutation
            # print(change.values.tolist()[0])
            # print(mutation.tolist())
            # print(final_change.values.tolist()[0])
            # print()
            new_company_values = [x + y for x, y in
                                  zip(new_companies[parent].to_array(), final_change.values.tolist()[0])]
            if not only_positive_values(new_company_values):
                continue
            predictions = self.structure_change_model.predict(final_change, verbose=None)
            prediction = predictions[0][0]
            child_company = Company(*new_company_values)
            if self.outliers_model.predict(child_company.to_dataframe())[0] == -1:
                continue
            child_company.change_company_value(prediction)
            new_companies[parent] = child_company
            i += 1

        # new_companies = sorted(new_companies, key=lambda company: company.value, reverse=True)[:self.mi]
        self.generated_companies = new_companies
