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
        while len(self.generated_companies) < number_of_companies:
            assets = generate_structure_mean(means[:5], 15)
            liabilities = generate_structure_mean(means[5:], 15)
            company = Company(*assets, *liabilities)
            if self.outliers_model.predict(company.to_dataframe())[0] == -1:
                continue
            self.generated_companies.append(company)
            with open(self.data_path, "a", newline="") as file:
                tmp = list([company.to_array()])
                writer = csv.writer(file)
                writer.writerows(tmp)
            print("Company structure generated. Total structures:", len(self.generated_companies))


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

    def display_metrics(self, metrics):
        for strategy_name, es_metrics in metrics.items():
            print(f"\nStrategia: {strategy_name}")
            print("=" * 80)

            value_metrics = es_metrics["value_metrics"]
            structure_metrics = es_metrics["structure_metrics"]
            time_metrics = es_metrics["time_metrics"]
            positive_changes = es_metrics["positive_changes"]

            print("Ilość pozytywnych zmian:", positive_changes)
            print("-" * 80)

            print("Czasowe statystyki:")
            for key, value in time_metrics.items():
                print(f"  {key}: {value:.4f}")
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
            medians = []

            for strategy_name, metrics in structure_metrics_per_strategy.items():
                per_feature = metrics["structure_metrics"]["per_feature_metrics"]
                if feature_name in per_feature:
                    strategies.append(strategy_name)
                    means.append(per_feature[feature_name]["mean_change"])
                    stds.append(per_feature[feature_name]["std_change"])
                    medians.append(per_feature[feature_name]["median_change"])

            if not strategies:
                continue  # Skip if no data for this feature

            plt.figure(figsize=(10, 6))
            plt.bar(strategies, means, yerr=stds, capsize=5, color="skyblue", edgecolor="black")
            x = np.arange(len(strategies))
            plt.scatter(x, medians, color="red", marker="D", label="Mediana zmian")
            plt.ylabel("Średnia zmiana")
            plt.title(f"Porównanie zmian: {feature_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()

    def draw_single_structures(self, es_structures, basic_structures, title, color):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        es_structures_2d = self.pca.transform(es_structures)
        basic_structures_2d = self.pca.transform(basic_structures)

        ax.scatter(basic_structures_2d[:, 0], basic_structures_2d[:, 1], c="#000000", edgecolors="#000000", s=4,
                   alpha=0.7, label="Bazowe")
        ax.scatter(es_structures_2d[:, 0], es_structures_2d[:, 1], c=color, edgecolors=color, s=4, alpha=0.7,
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

    def plot_value_vs_time(self, metrics):
        strategies = []
        efficiencies = []

        for strategy_name, data in metrics.items():
            mean_increase = data["value_metrics"]["mean_increase_percentage"]
            total_time = data["time_metrics"]["sum_time"]

            if total_time > 0:
                efficiency = mean_increase / total_time
            else:
                efficiency = 0

            strategies.append(strategy_name)
            efficiencies.append(efficiency)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, efficiencies, color="mediumseagreen", edgecolor="black")
        plt.ylabel("Efektywność [% wzrostu / s]")
        plt.title("Efektywność strategii ewolucyjnych")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2.0, height), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_time_metrics(self, metrics):
        strategy_names = []
        total_times = []

        for strategy_name, strategy_metrics in metrics.items():
            strategy_names.append(strategy_name)
            total_times.append(strategy_metrics["time_metrics"]["sum_time"])

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategy_names, total_times, color="lightcoral", edgecolor="black")
        plt.ylabel("Czas całkowity [s]")
        plt.title("Porównanie całkowitego czasu działania strategii")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        # Add time labels on top of each bar
        for bar, time in zip(bars, total_times):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_value_change_comparison(self, metrics):
        strategy_names = []
        means = []
        for strategy_name, strategy_metrics in metrics.items():
            strategy_names.append(strategy_name)
            means.append(strategy_metrics["value_metrics"]["mean_increase_percentage"])

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategy_names, means, color="lightcoral", edgecolor="black")
        plt.ylabel("Średnie zmiany wartości przedsiębiorstw [%]")
        plt.title("Porównanie średnich zmian wartości przedsiębiorstw")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        for bar, time in zip(bars, means):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.2f}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def plot_positive_changes_comparison(self, metrics):
        strategy_names = []
        total_positives = []
        for strategy_name, strategy_metrics in metrics.items():
            strategy_names.append(strategy_name)
            total_positives.append(strategy_metrics["positive_changes"])

        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategy_names, total_positives, color="lightcoral", edgecolor="black")
        plt.ylabel("Ilość pozytywnych zmian wartości przedsiębiorstw")
        plt.title("Porównanie ilości pozytywnych zmian wartości przedsiębiorstw")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        for bar, time in zip(bars, total_positives):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()