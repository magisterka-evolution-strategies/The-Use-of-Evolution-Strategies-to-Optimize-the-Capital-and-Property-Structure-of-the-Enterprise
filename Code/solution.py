import numpy as np
from sklearn.decomposition import PCA

from Code.utils.data_visualization import visualize_all
from Outliers import Outliers
from StructureChange import StructureChange
from Code.utils.retrieve_data import get_raw_sql_data
from Code.utils.data_modification import get_percentage_structure, get_structure_changes, get_filtered_changes

filename = "exporter_new.db"

data = get_raw_sql_data(filename)

print(len(data))
print(data[0])

# company_id, period, market_value, 5 x assets, 5 x liabilities
percentage_structure_data = get_percentage_structure(data)

print(len(percentage_structure_data))
print(percentage_structure_data[0])

outliers = Outliers(percentage_structure_data)
isolation_forest = outliers.train_model(contamination=0.034)
# outliers.check(percentage_structure_data)
# means = outliers.get_mean_values()

# structure_changes_data = get_structure_changes(percentage_structure_data)
#
# filtered_changes_data = get_filtered_changes(structure_changes_data)
#
# structure_change = StructureChange(filtered_changes_data)
# structure_change_model = structure_change.train_model()

# number_of_companies = 10
# evolutionary_algorithm = EvolutionaryAlgorithm(number_of_companies, means, isolation_forest, structure_change_model)
#
# evolutionary_algorithm.check_generated_structures()
#
# epochs = 100
# for i in range(epochs):
#     evolutionary_algorithm.generate_offspring()
#
# evolutionary_algorithm.check_generated_structures()
#
