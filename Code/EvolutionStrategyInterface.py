import copy
from typing import List

import numpy as np

from Code import EvolutionPlatform
from Code.Company import Company


class EvolutionStrategyInterface:
    def __init__(self, evolution_platform: EvolutionPlatform, name: str):
        self.evolution_platform = evolution_platform
        self.generated_companies: List[Company] = copy.deepcopy(evolution_platform.generated_companies)
        self.outliers_model = evolution_platform.outliers_model
        self.structure_change_model = evolution_platform.structure_change_model
        self.name = name
        self.positive_changes_made = 0
        self.evaluation_times = []

    def check_generated_structures(self):
        for company in self.generated_companies:
            values = company.to_dataframe().values[0]
            print("{:.2f}".format(company.value), ["{:.2f}".format(num) for num in values])

    def calculate_value_increase_metrics(self):
        initial_values = np.array([100 for _ in self.generated_companies])
        final_values = np.array([company.value for company in self.generated_companies])

        value_increase_percentage = (final_values - initial_values) / initial_values * 100

        metrics = {
            "mean_increase_percentage": np.mean(value_increase_percentage),
            "std_increase_percentage": np.std(value_increase_percentage),
            "median_increase_percentage": np.median(value_increase_percentage),
            "min_increase_percentage": np.min(value_increase_percentage),
            "max_increase_percentage": np.max(value_increase_percentage),
        }

        return metrics

    def calculate_structure_change_metrics(self):
        all_changes = []

        for index, company in enumerate(self.generated_companies):
            initial_structure = self.evolution_platform.generated_companies[index].to_dataframe()
            final_structure = company.to_dataframe()

            change = (final_structure.values - initial_structure.values).flatten()

            all_changes.append(change)
            # change_abs = np.abs(change)
            #
            # all_changes.append(change_abs)

        all_changes = np.array(all_changes)

        # mean_change_per_company = np.mean(all_changes, axis=1)

        # overall_metrics = {
        #     "mean_structure_change_percentage": np.mean(mean_change_per_company),
        #     "std_structure_change_percentage": np.std(mean_change_per_company),
        #     "median_structure_change_percentage": np.median(mean_change_per_company),
        #     "min_structure_change_percentage": np.min(mean_change_per_company),
        #     "max_structure_change_percentage": np.max(mean_change_per_company),
        # }

        columns = [
            "NonCurrentAssets", "CurrentAssets", "AssetsHeldForSaleAndDiscountinuingOperations",
            "CalledUpCapital", "OwnShares", "EquityShareholdersOfTheParent",
            "NonControllingInterests", "NonCurrentLiabilities", "CurrentLiabilities",
            "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"
        ]

        per_feature_metrics = {}

        for idx, col in enumerate(columns):
            feature_changes = all_changes[:, idx]

            per_feature_metrics[col] = {
                "mean_change": np.mean(feature_changes),
                "std_change": np.std(feature_changes),
                "median_change": np.median(feature_changes),
                "min_change": np.min(feature_changes),
                "max_change": np.max(feature_changes),
            }

        return {
            # "overall_metrics": overall_metrics,
            "per_feature_metrics": per_feature_metrics
        }

    def calculate_time_metrics(self):
        values = np.array(self.evaluation_times)

        metrics = {
            "mean_time": np.mean(values),
            "sum_time": np.sum(values)
        }

        return metrics

    def generate_offspring(self):
        pass
