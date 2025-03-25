import numpy as np

from Code.utils.data_modification import get_percentage_structure, get_structure_changes
from Code.utils.retrieve_data import get_raw_sql_data

filename = "../exporter_new.db"

data = get_raw_sql_data(filename)

print(data[0])
print(data[1])
print(len(data))

# company_id, period, market_value, 5 x assets, 5 x liabilities
percentage_structure_data = get_percentage_structure(data)

print(percentage_structure_data[0])
print(percentage_structure_data[1])
print(len(percentage_structure_data))

final_data = get_structure_changes(percentage_structure_data)

print(final_data[0])
print(final_data[1])
print(len(final_data))

only_structures = [structure[3:] for structure in final_data]

print(only_structures[0])
print(only_structures[1])
print(len(only_structures))

only_structures = np.array(only_structures)

left_half = only_structures[:, :5]  # First 5 columns
right_half = only_structures[:, 5:]  # Last 5 columns

print(left_half[0], right_half[0])
print(left_half[1], right_half[1])

left_sum = np.sum(np.abs(left_half), axis=1)
right_sum = np.sum(np.abs(right_half), axis=1)

print(left_sum)
print(right_sum)

left_mean = np.mean(left_sum)
right_mean = np.mean(right_sum)

print("Mean of left half sums:", left_mean)
print("Mean of right half sums:", right_mean)

left_median = np.median(left_sum)
right_median = np.median(right_sum)

print("Median of left half sums:", left_median)
print("Median of right half sums:", right_median)