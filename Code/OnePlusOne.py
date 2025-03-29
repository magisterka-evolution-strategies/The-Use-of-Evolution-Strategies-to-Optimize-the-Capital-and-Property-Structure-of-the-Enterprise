import random

import pandas as pd

from Code.Company import Company
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface


class OnePlusOne(EvolutionStrategyInterface):
    def __init__(self, evolution_platform, means_changes):
        super().__init__(evolution_platform)
        self.means_changes = means_changes

    def generate_gradient(self):
        values = []
        for _ in range(4):
            values.append(random.uniform(-1, 1))
        fifth_value = -sum(values)
        if -1 <= fifth_value <= 1:
            values.append(fifth_value)
        else:
            return self.generate_gradient()
        return values

    def generate_offspring(self):
        new_companies = []
        for company in self.generated_companies:
            gradient_assets = self.generate_gradient()
            gradient_liabilities = self.generate_gradient()
            df = pd.DataFrame([[*gradient_assets, *gradient_liabilities]],
                              columns=["NonCurrentAssets", "CurrentAssets",
                                       "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                       "EquityShareholdersOfTheParent", "NonControllingInterests",
                                       "NonCurrentLiabilities",
                                       "CurrentLiabilities",
                                       "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])
            predictions = self.structure_change_model.predict(df)
            prediction = predictions[0][0]
            if prediction > 0:
                new_company_values = [x + y for x, y in
                                      zip(company.to_array(), [*gradient_assets, *gradient_liabilities])]
                child_company = Company(*new_company_values)
                if self.outliers_model.predict(child_company.to_dataframe())[0] != -1:
                    new_companies.append(child_company)
                    continue
            new_companies.append(company)

        self.generated_companies = new_companies
