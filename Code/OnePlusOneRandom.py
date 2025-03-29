import random

import pandas as pd

from Code.Company import Company
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class OnePlusOneRandom(EvolutionStrategyInterface):
    def __init__(self, evolution_platform):
        super().__init__(evolution_platform)

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

    def generate_offspring(self):
        new_companies = []
        for i, company in enumerate(self.generated_companies):
            while i + 1 != len(new_companies):
                gradient_assets = self.generate_random_gradient()
                gradient_liabilities = self.generate_random_gradient()
                new_company_values = [x + y for x, y in
                                      zip(company.to_array(), [*gradient_assets, *gradient_liabilities])]
                if not only_positive_values(new_company_values):
                    continue
                df = pd.DataFrame([[*gradient_assets, *gradient_liabilities]],
                                  columns=["NonCurrentAssets", "CurrentAssets",
                                           "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                           "EquityShareholdersOfTheParent", "NonControllingInterests",
                                           "NonCurrentLiabilities",
                                           "CurrentLiabilities",
                                           "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])
                predictions = self.structure_change_model.predict(df, verbose=None)
                prediction = predictions[0][0]
                if prediction <= 0:
                    continue
                child_company = Company(*new_company_values)
                if self.outliers_model.predict(child_company.to_dataframe())[0] == -1:
                    continue
                new_companies.append(child_company)

        self.generated_companies = new_companies
