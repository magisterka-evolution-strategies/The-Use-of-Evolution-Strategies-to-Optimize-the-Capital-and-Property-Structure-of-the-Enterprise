import random

import numpy as np
import pandas as pd

from Code.Company import Company
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class OnePlusOneMean(EvolutionStrategyInterface):
    def __init__(self, evolution_platform, mean_changes, std_changes):
        super().__init__(evolution_platform)
        self.mean_assets = mean_changes[:5]
        self.mean_liabilities = mean_changes[5:]
        self.std_assets = std_changes[:5]
        self.std_liabilities = std_changes[5:]

    def generate_assets_gradient(self):
        n = len(self.mean_assets)

        gradients = np.random.normal(loc=self.mean_assets, scale=self.std_assets, size=n)

        gradients -= np.mean(gradients)

        return gradients.tolist()

    def generate_liabilities_gradient(self):
        n = len(self.mean_liabilities)

        gradients = np.random.normal(loc=self.mean_liabilities, scale=self.std_liabilities, size=n)

        gradients -= np.mean(gradients)

        return gradients.tolist()

    def generate_offspring(self):
        new_companies = []
        for i, company in enumerate(self.generated_companies):
            while i + 1 != len(new_companies):
                gradient_assets = self.generate_assets_gradient()
                gradient_liabilities = self.generate_liabilities_gradient()
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
