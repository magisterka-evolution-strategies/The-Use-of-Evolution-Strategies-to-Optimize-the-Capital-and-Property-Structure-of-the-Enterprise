import csv
import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from Code.EvolutionStrategyInterface import EvolutionStrategyInterface
from Company import Company
from Code.utils.generate_company_structure import generate_structure_mean


class EvolutionPlatform:
    def __init__(self, outliers_model, structure_change_model):
        self.outliers_model = outliers_model
        self.structure_change_model = structure_change_model
        self.generated_companies: List[Company] = []
        self.evolution_strategies: List[EvolutionStrategyInterface] = []

        self.data_path = "data/start_companies.csv"
        self.pca = PCA(n_components=2, random_state=42)

    def generate_start_companies(self, number_of_companies, means):
        start_len = len(self.generated_companies)

        while len(self.generated_companies) < number_of_companies:
            assets = generate_structure_mean(means[:5], 10)
            liabilities = generate_structure_mean(means[5:], 10)
            company = Company(*assets, *liabilities)
            if self.outliers_model.predict(company.to_dataframe())[0] == -1:
                continue
            self.generated_companies.append(company)
            print("Company structure generated. Total structures:", len(self.generated_companies))

        if start_len >= number_of_companies:
            return
        with open(self.data_path, "a", newline="") as file:
            tmp = list(map(lambda x: x.to_array(), self.generated_companies[start_len:]))
            writer = csv.writer(file)
            writer.writerows(tmp)

    def load_companies(self, number_of_companies, means):
        if not os.path.exists(self.data_path):
            return
        with open(self.data_path, "r") as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            tmp = [row for row in reader]
            tmp = tmp[:min(number_of_companies, len(tmp))]
            for structure in tmp:
                company = Company(*structure)
                self.generated_companies.append(company)
        self.generate_start_companies(number_of_companies, means)

    def fit_visualization(self):
        data = np.array([company.to_array() for company in self.generated_companies])
        self.pca.fit_transform(data)

    def show_structures(self):
        for company in self.generated_companies:
            values = company.to_dataframe().values[0]
            print(["{:.2f}".format(num) for num in values])

    def show_all(self):
        for evolution_strategy in self.evolution_strategies:
            print(evolution_strategy.name, evolution_strategy.positive_changes_made)
            evolution_strategy.check_generated_structures()

    def show_all(self):
        for evolution_strategy in self.evolution_strategies:
            print(evolution_strategy.name)
            evolution_strategy.check_generated_structures()

    def add_evolution_strategy(self, evolution_strategy: EvolutionStrategyInterface):
        self.evolution_strategies.append(evolution_strategy)

    def start_evolution(self, epochs: int):
        for epoch in range(epochs):
            print("Epoch: {0}/{1}".format(epoch + 1, epochs))
            for i, evolution_strategy in enumerate(self.evolution_strategies):
                evolution_strategy.generate_offspring()
                print("Complete: {0}/{1}".format(i + 1, len(self.evolution_strategies)))

    def plot_structure_changes(self, structure_metrics_per_strategy):
        features = [
            "NonCurrentAssets", "CurrentAssets", "AssetsHeldForSaleAndDiscountinuingOperations",
            "CalledUpCapital", "OwnShares", "EquityShareholdersOfTheParent",
            "NonControllingInterests", "NonCurrentLiabilities", "CurrentLiabilities",
            "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"
        ]

        for feature_name in features:
            strategies = []
            means = []
            stds = []

            for strategy_name, metrics in structure_metrics_per_strategy.items():
                per_feature = metrics["structure_metrics"]["per_feature_metrics"]
                if feature_name in per_feature:
                    strategies.append(strategy_name)
                    means.append(per_feature[feature_name]["mean_change"])
                    stds.append(per_feature[feature_name]["std_change"])

            if not strategies:
                continue  # Skip if no data for this feature

            plt.figure(figsize=(10, 6))
            plt.bar(strategies, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
            plt.ylabel("Średnia zmiana")
            plt.title(f"Porównanie zmian: {feature_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def display_metrics(self, metrics):
        for strategy_name, es_metrics in metrics.items():
            print(f"\nStrategia: {strategy_name}")
            print("=" * 80)

            value_metrics = es_metrics["value_metrics"]
            structure_metrics = es_metrics["structure_metrics"]
            positive_changes = es_metrics["positive_changes"]

            print("Ilość pozytywnych zmian:", positive_changes)
            print("-" * 80)

            print("Wzrost wartości przedsiębiorstwa:")
            for key, value in value_metrics.items():
                print(f"  {key}: {value:.4f}")
            print("-" * 80)

            # print("Zagregowane zmiany struktury kapitałowej:")
            # for key, value in structure_metrics["overall_metrics"].items():
            #     print(f"  {key}: {value:.4f}")

            print("Szczegółowe zmiany dla poszczególnych składowych:")
            for feature, stats in structure_metrics["per_feature_metrics"].items():
                print(f"  {feature}:")
                for stat_name, stat_value in stats.items():
                    print(f"    {stat_name}: {stat_value:.4f}")
                print("-" * 40)

            print("=" * 80)

    def calculate_metrics(self):
        metrics = {}
        for evolution_strategy in self.evolution_strategies:
            value_metrics = evolution_strategy.calculate_value_increase_metrics()
            structure_metrics = evolution_strategy.calculate_structure_change_metrics()
            time_metrics = evolution_strategy.calculate_time_metrics()

            es_metrics = {
                "value_metrics": value_metrics,
                "structure_metrics": structure_metrics,
                "time_metrics": time_metrics,
                "positive_changes": evolution_strategy.positive_changes_made
            }
            metrics[evolution_strategy.name] = es_metrics

        return metrics

    def draw_single_structures(self, es_structures, basic_structures, title, color):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        es_structures_2d = self.pca.transform(es_structures)
        basic_structures_2d = self.pca.transform(basic_structures)

        ax.scatter(basic_structures_2d[:, 0], basic_structures_2d[:, 1], c="#000000", edgecolors="#000000", s=2,
                   alpha=0.7, label="Bazowe")
        ax.scatter(es_structures_2d[:, 0], es_structures_2d[:, 1], c=color, edgecolors=color, s=2, alpha=0.7,
                   label=title)

        plt.xlabel("Wymiar 1 (PCA)")
        plt.ylabel("Wymiar 2 (PCA)")
        plt.title("Wizualizacja struktur kapitałowych przedsiębiorstw")
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_total_structures(self, basic_structures, total_structures):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        basic_structures_2d = self.pca.transform(basic_structures)
        ax.scatter(basic_structures_2d[:, 0], basic_structures_2d[:, 1], c="#000000", edgecolors="#000000", s=2,
                   alpha=0.7, label="Bazowe")

        for es_structures, label in total_structures:
            es_structures_2d = self.pca.transform(es_structures)
            ax.scatter(es_structures_2d[:, 0], es_structures_2d[:, 1], s=2, alpha=0.7, label=label)

        plt.xlabel("Wymiar 1 (PCA)")
        plt.ylabel("Wymiar 2 (PCA)")
        plt.title("Ogólna wizualizacja struktur kapitałowych przedsiębiorstw")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_structure_changes(self):
        basic_structures = np.array([company.to_array() for company in self.generated_companies])
        total_structures = []
        for evolution_strategy in self.evolution_strategies:
            es_structures = np.array([company.to_array() for company in evolution_strategy.generated_companies])
            total_structures.append((es_structures, evolution_strategy.name))
            self.draw_single_structures(es_structures, basic_structures, evolution_strategy.name, "#ff0000")

        self.draw_total_structures(basic_structures, total_structures)
