import matplotlib.pyplot as plt


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

    ax.scatter(all_data_2d[:, 0], all_data_2d[:, 1], c=color, edgecolors=color, s=2, alpha=0.7, label="Anomalie")

    ax.scatter(filtered_data_2d[:, 0], filtered_data_2d[:, 1], c="#000000", edgecolors="#000000", s=2, alpha=0.7, label="Dane")

    plt.xlabel("Wymiar 1 (PCA)")
    plt.ylabel("Wymiar 2 (PCA)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
