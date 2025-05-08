import math

import numpy as np
import pandas as pd
from pandas import Series
from pymoo.core.problem import Problem

from Code.Company import Company
from Code.utils.calculations import only_positive_values


class Pymoo(Problem):
    def __init__(self, companies, mean_changes: Series | float, std_changes: Series | float, structure_change_model, outliers_model):
        super().__init__(n_var=10, n_obj=1, xl=-1, xu=1)
        self.base_companies = companies
        self.mean_changes = mean_changes
        self.std_changes = std_changes
        self.structure_change_model = structure_change_model
        self.outliers_model = outliers_model
        self.changes_to_apply = []

    def _evaluate(self, X, out, *args, **kwargs):
        fitness = []
        self.changes_to_apply = []

        for i, change in enumerate(X):
            company_index = i % len(self.base_companies)
            base_company = self.base_companies[company_index]

            # raw_change = np.random.normal(loc=self.mean_changes, scale=self.std_changes)
            raw_change = np.random.normal(loc=self.mean_changes, scale=0.1)
            change = change * raw_change

            part1 = change[:5]
            part1 -= np.mean(part1)
            part2 = change[5:]
            part2 -= np.mean(part2)
            change = np.concatenate([part1, part2])

            df = pd.DataFrame([change], columns=[
                "NonCurrentAssets", "CurrentAssets", "AssetsHeldForSaleAndDiscountinuingOperations",
                "CalledUpCapital", "OwnShares", "EquityShareholdersOfTheParent",
                "NonControllingInterests", "NonCurrentLiabilities", "CurrentLiabilities",
                "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"
            ])

            prediction = self.structure_change_model.predict(df, verbose=None)[0][0]
            predicted_value = base_company.value * (100 + prediction) / 100

            fitness.append(-predicted_value)

            self.changes_to_apply.append((company_index, change, prediction))

        out["F"] = np.array(fitness).reshape(-1, 1)

    def apply_best_changes(self):
        best_changes = {}

        for company_index, company in enumerate(self.base_companies):
            best_changes[company_index] = (None, -math.inf)

        for (company_index, change, prediction) in self.changes_to_apply:
            if prediction > best_changes[company_index][1]:
                new_company_values = np.array(self.base_companies[company_index].to_array()) + change.ravel()
                if not only_positive_values(new_company_values):
                    continue
                child_company = Company(*new_company_values)
                if self.outliers_model.predict(child_company.to_dataframe())[0] == -1:
                    continue
                best_changes[company_index] = (change, prediction)

        positive_changes = 0
        for company_index, (change, prediction) in best_changes.items():
            if prediction == -math.inf:
                continue
            # print(company_index, prediction, change)
            if prediction > 0:
                positive_changes += 1
            self.base_companies[company_index].modify_structure(*change.ravel())
            self.base_companies[company_index].change_company_value(prediction)

        return positive_changes
