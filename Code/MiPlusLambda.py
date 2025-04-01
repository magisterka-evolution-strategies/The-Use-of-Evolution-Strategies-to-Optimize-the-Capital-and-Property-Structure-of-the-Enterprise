import random

import numpy as np
import pandas as pd

from Code.Company import Company
from Code.EvolutionPlatform import EvolutionPlatform
from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Code.utils.calculations import only_positive_values


class MiPlusLambda(EvolutionStrategyInterface):
    def __init__(self, evolution_platform: EvolutionPlatform, mi: int, la: int ):
        super().__init__(evolution_platform)
        self.mi = mi
        self.la = la

    def generate_possible_changes(self, company: Company):
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
            change = (company.to_dataframe() - df.mean()).values.tolist()[0]
            print("{0}: {1}".format(i, change))

    def generate_offspring(self):
        new_companies = []
        for i, company in enumerate(self.generated_companies):
            while i + 1 != len(new_companies):
                print("CURR {0}".format(company.to_array()))
                self.generate_possible_changes(company)
                # gradient_assets = self.generate_assets_gradient()
                # gradient_liabilities = self.generate_liabilities_gradient()
                # new_company_values = [x + y for x, y in
                #                       zip(company.to_array(), [*gradient_assets, *gradient_liabilities])]
                # if not only_positive_values(new_company_values):
                #     continue
                # df = pd.DataFrame([[*gradient_assets, *gradient_liabilities]],
                #                   columns=["NonCurrentAssets", "CurrentAssets",
                #                            "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                #                            "EquityShareholdersOfTheParent", "NonControllingInterests",
                #                            "NonCurrentLiabilities",
                #                            "CurrentLiabilities",
                #                            "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])
                # predictions = self.structure_change_model.predict(df, verbose=None)
                # prediction = predictions[0][0]
                # if prediction <= 0:
                #     continue
                # child_company = Company(*new_company_values)
                # if self.outliers_model.predict(child_company.to_dataframe())[0] == -1:
                #     continue
                child_company = self.generated_companies[i]
                new_companies.append(child_company)

        self.generated_companies = new_companies
