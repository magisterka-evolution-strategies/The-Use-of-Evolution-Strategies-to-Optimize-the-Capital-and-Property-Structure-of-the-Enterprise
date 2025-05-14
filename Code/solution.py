from Code.EvolutionPlatform import EvolutionPlatform
from Code.MiCommaLambda import MiCommaLambda
from Code.MiPlusLambda import MiPlusLambda
from Code.OnePlusOneMean import OnePlusOneMean
from Code.OnePlusOneRandom import OnePlusOneRandom
from Code.PymooES import PymooES
from Code.utils.data_information import get_structure_data_statistics, get_change_data_statistics
from Outliers import Outliers
from StructureChange import StructureChange
from Code.utils.retrieve_data import get_raw_sql_data
from Code.utils.data_modification import get_percentage_structure, get_structure_changes, get_filtered_changes, \
    adjust_std_based_on_mean

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

mean_structures, std_structures = get_structure_data_statistics(percentage_structure_data, isolation_forest)
# get_data_statistics(filtered_changes_data)
mean_changes, std_changes = get_change_data_statistics(filtered_changes_data)
adjusted_std = adjust_std_based_on_mean(mean_changes, std_changes)

structure_change = StructureChange(filtered_changes_data)
structure_change_model = structure_change.get_model()

evolution_platform = EvolutionPlatform(isolation_forest, structure_change_model)

number_of_companies = 100
evolution_platform.load_companies(number_of_companies, mean_structures)
# evolution_platform.fit_visualization()
#
# evolution_platform.show_structures()
#
# one_plus_one_random = OnePlusOneRandom(evolution_platform, "1+1 losowe")
# one_plus_one_mean_normal = OnePlusOneMean(evolution_platform, "1+1 rozkład normalny", mean_changes, std_changes)
# one_plus_one_mean_adjusted = OnePlusOneMean(evolution_platform, "1+1 rozkład normalny (std)", mean_changes, adjusted_std)
# mi = 10
# la = 5
# factor = 8
# mi_plus_lambda = MiPlusLambda(evolution_platform, r"$\mu$+$\lambda$", mi, la, factor)
# mi_comma_lambda = MiCommaLambda(evolution_platform, r"$\mu$,$\lambda$", mi, la, factor)
# pymoo_es_normal = PymooES(evolution_platform, "Pymoo", mean_changes, std_changes)
# pymoo_es_adjusted = PymooES(evolution_platform, "Pymoo (std)", mean_changes, adjusted_std)
#
# evolution_platform.add_evolution_strategy(one_plus_one_random)
# evolution_platform.add_evolution_strategy(one_plus_one_mean_normal)
# evolution_platform.add_evolution_strategy(one_plus_one_mean_adjusted)
# evolution_platform.add_evolution_strategy(mi_plus_lambda)
# evolution_platform.add_evolution_strategy(mi_comma_lambda)
# evolution_platform.add_evolution_strategy(pymoo_es_normal)
# evolution_platform.add_evolution_strategy(pymoo_es_adjusted)
#
# epochs = 2
# evolution_platform.start_evolution(epochs)
#
# evolution_platform.show_all()
#
# metrics = evolution_platform.calculate_metrics()
#
# evolution_platform.display_metrics(metrics)
#
# evolution_platform.visualize_structure_changes()
#
# evolution_platform.plot_structure_changes(metrics)
#
# evolution_platform.plot_time_metrics(metrics)
# evolution_platform.plot_value_vs_time(metrics)
#
# evolution_platform.plot_value_change_comparison(metrics)
#
# evolution_platform.plot_positive_changes_comparison(metrics)
