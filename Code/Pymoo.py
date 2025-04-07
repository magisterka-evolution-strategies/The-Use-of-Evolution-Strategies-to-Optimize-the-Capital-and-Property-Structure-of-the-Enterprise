import math

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from Code.utils.calculations import only_positive_values


class Pymoo(Problem):
    def __init__(self, companies, ml_model):
        super().__init__(n_var=10, n_obj=1, xl=-2, xu=2)
        self.base_companies = companies
        self.ml_model = ml_model
        self.changes_to_apply = []

    def _evaluate(self, X, out, *args, **kwargs):
        fitness = []
        self.changes_to_apply = []

        for i, change in enumerate(X):
            company_index = i % len(self.base_companies)
            base_company = self.base_companies[company_index]

            df = pd.DataFrame([change], columns=[
                "NonCurrentAssets", "CurrentAssets", "AssetsHeldForSaleAndDiscountinuingOperations",
                "CalledUpCapital", "OwnShares", "EquityShareholdersOfTheParent",
                "NonControllingInterests", "NonCurrentLiabilities", "CurrentLiabilities",
                "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"
            ])

            prediction = self.ml_model.predict(df, verbose=None)[0][0]
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
                best_changes[company_index] = (change, prediction)

        for company_index, (change, prediction) in best_changes.items():
            print(company_index, prediction)
            if prediction == -math.inf:
                continue
            self.base_companies[company_index].modify_structure(*change.ravel())
            self.base_companies[company_index].change_company_value(prediction)
