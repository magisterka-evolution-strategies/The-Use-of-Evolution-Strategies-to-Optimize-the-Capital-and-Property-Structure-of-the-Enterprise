import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from Code.utils.calculations import percentage
from Code.utils.data_modification import get_percentage_structure, get_structure_changes
from Code.utils.retrieve_data import get_raw_sql_data


def visualize(visualization_data, title, color, pca, xlim, ylim):
    data_2d = pca.transform(visualization_data)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(data_2d[:, 0], data_2d[:, 1], c=color, edgecolors=color, s=1, alpha=0.7)

    plt.xlabel("Wymiar 1 (PCA)")
    plt.ylabel("Wymiar 2 (PCA)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_all(all_data, filtered_data, title, color, pca, xlim, ylim):
    all_data_2d = pca.transform(all_data)
    filtered_data_2d = pca.transform(filtered_data)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(all_data_2d[:, 0], all_data_2d[:, 1], c=color, edgecolors=color, s=2, alpha=0.7, label="Wszystkie dane")

    ax.scatter(filtered_data_2d[:, 0], filtered_data_2d[:, 1], c="#000000", edgecolors="#000000", s=2, alpha=0.7, label="Przefiltrowane dane")

    plt.xlabel("Wymiar 1 (PCA)")
    plt.ylabel("Wymiar 2 (PCA)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

filename = "../exporter_new.db"

data = get_raw_sql_data(filename)

text = "Wizualizacja struktur kapitałowych przedsiębiorstw"
color = "#b52264"
percentage_structure_data = get_percentage_structure(data)

pca = PCA(n_components=2, random_state=42)
data_2d = pca.fit_transform(np.array(np.stack(percentage_structure_data)[:, 3:]))

only_structure = np.array(np.stack(percentage_structure_data)[:, 3:], dtype=float)

data_standardized = zscore(only_structure)

mask = np.abs(data_standardized) < 3
data_filtered = only_structure[np.all(mask, axis=1)]

print(len(only_structure))
print(len(data_filtered))
print(100 - percentage(len(data_filtered), len(only_structure)))

visualize(np.array(np.stack(percentage_structure_data)[:, 3:], dtype=float), text, color, pca, [-120, 80], [-100, 100])

visualize(data_filtered, text, color, pca, [-120, 80], [-100, 100])

visualize_all(only_structure, data_filtered, text, color, pca, [-120, 80], [-100, 100])



text = "Wizualizacja zmian struktur kapitałowych przedsiębiorstw"
color = "#e83c3c"
final_data = get_structure_changes(percentage_structure_data)

pca = PCA(n_components=2, random_state=42)
data_2d = pca.fit_transform(np.array(np.stack(final_data)[:, 3:]))

only_structure = np.array(np.stack(final_data)[:, 3:], dtype=float)

data_standardized = zscore(only_structure)

mask = np.abs(data_standardized) < 3
data_filtered = only_structure[np.all(mask, axis=1)]

print(len(only_structure))
print(len(data_filtered))
print(100 - percentage(len(data_filtered), len(only_structure)))

visualize(np.array(np.stack(final_data)[:, 3:], dtype=float), text, color, pca, [-120, 120], [-120, 160])

visualize(data_filtered, text, color, pca, [-120, 120], [-120, 160])

visualize_all(only_structure, data_filtered, text, color, pca, [-120, 120], [-120, 160])








# Q1 = np.percentile(x, 25, axis=0)
# Q3 = np.percentile(x, 75, axis=0)
# IQR = Q3 - Q1
# mask = (x >= Q1 - 1.5 * IQR) & (x <= Q3 + 1.5 * IQR)
# data_filtered = x[np.all(mask, axis=1)]
#
# print(len(data_filtered))
#
# # visualize(data_filtered, "Wizualizacja struktur kapitałowych przedsiębiorstw3", "#222a9c")
#
#
#
#
#
# from sklearn.neighbors import LocalOutlierFactor
# lof = LocalOutlierFactor(n_neighbors=20)
# outlier_mask = lof.fit_predict(x)  # -1 oznacza outlierów
# data_filtered = x[outlier_mask == 1]  # Usuwamy outlierów
#
# print(len(data_filtered))
#
# # visualize(data_filtered, "Wizualizacja struktur kapitałowych przedsiębiorstw4", "#222a9c")