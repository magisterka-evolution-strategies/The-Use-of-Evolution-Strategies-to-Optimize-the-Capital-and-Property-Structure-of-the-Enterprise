from Code.EvolutionPlatform import EvolutionPlatform
from Code.utils.data_information import get_data_statistics
from Code.utils.data_visualization import visualize_all
from Outliers import Outliers
from StructureChange import StructureChange
from Code.utils.retrieve_data import get_raw_sql_data
from Code.utils.data_modification import get_percentage_structure, get_structure_changes, get_filtered_changes

filename = "exporter_new.db"

data = get_raw_sql_data(filename)

# company_id, period, market_value, 5 x assets, 5 x liabilities
percentage_structure_data = get_percentage_structure(data)

outliers = Outliers(percentage_structure_data)
isolation_forest = outliers.get_model()
# isolation_forest = outliers.train_model(contamination=0.034)
# outliers.check(percentage_structure_data)
# means = outliers.get_mean_values()

structure_changes_data = get_structure_changes(percentage_structure_data)

filtered_changes_data = get_filtered_changes(structure_changes_data)

means = get_data_statistics(percentage_structure_data, isolation_forest)
# get_data_statistics(filtered_changes_data)

structure_change = StructureChange(filtered_changes_data)
structure_change_model = structure_change.get_model()

evolution_platform = EvolutionPlatform(isolation_forest, structure_change_model)

number_of_companies = 10
evolution_platform.load_companies(number_of_companies)
evolution_platform.generate_start_companies(number_of_companies, means)

evolution_platform.check_generated_structures()
