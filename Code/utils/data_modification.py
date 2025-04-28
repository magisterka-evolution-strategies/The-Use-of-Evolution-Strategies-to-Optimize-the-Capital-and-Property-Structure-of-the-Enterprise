import numpy as np
import pandas as pd
from scipy.stats import zscore

from Code.utils.calculations import percentage, only_positive_values


def get_percentage_structure(data):
    percentage_structure_data = []

    for i in range(len(data)):
        company_id = data[i][0]
        period = data[i][1]
        market_value = data[i][2]
        assets = data[i][3:8]
        liabilities = data[i][8:13]
        total_assets = sum(assets)
        total_liabilities = sum(liabilities)

        row = [company_id, period, market_value]

        for asset in assets:
            row.append(percentage(asset, total_assets))
        for liability in liabilities:
            row.append(percentage(liability, total_liabilities))

        if only_positive_values(assets) and only_positive_values(liabilities):  # if we accept only positive values
            percentage_structure_data.append(row)

    return percentage_structure_data


def get_structure_changes(data):
    final_data = []

    for i in range(1, len(data)):
        if data[i][0] != data[i - 1][0]:
            continue
        x = [data[i][0], data[i][1],
             percentage((data[i][2] - data[i - 1][2]), data[i - 1][2])]
        # x = [data[i][0], data[i][1], 1 if data[i][2] - data[i - 1][2] > 0 else 0]
        for j in range(3, len(data[i])):
            x.append(data[i][j] - data[i - 1][j])
            # if data[i - 1][j] != 0:
            #     x.append(percentage(data[i][j] - data[i - 1][j], data[i - 1][j]))
            # else:
            #     x.append(data[i][j])
        final_data.append(x)

    return final_data


def get_structure_changes_all(data):
    final_data = []

    for i in range(1, len(data)):
        for j in range(i - 1, 0, -1):
            if data[i][0] != data[j][0]:
                break
            x = [data[i][0], data[i][1],
                 percentage((data[i][2] - data[j][2]), data[j][2])]
            for k in range(3, len(data[i])):
                x.append(data[i][k] - data[j][k])
            final_data.append(x)

    return final_data


def get_filtered_changes(data):
    final_data_np = np.array(data, dtype=object)
    only_structure = np.array(final_data_np[:, 3:], dtype=float)
    data_standardized = zscore(only_structure)
    mask = np.abs(data_standardized) < 3
    valid_rows = np.all(mask, axis=1)
    filtered_data = final_data_np[valid_rows]

    return filtered_data.tolist()


def adjust_std_based_on_mean(mean_changes, std_changes):
    adjusted_std = np.where(abs(mean_changes) > 0.025, std_changes, 0.1)
    return adjusted_std
