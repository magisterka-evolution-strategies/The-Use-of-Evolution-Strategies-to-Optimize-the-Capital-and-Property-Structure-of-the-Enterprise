from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from Code.EvolutionPlatform import EvolutionPlatform
from Code.MiCommaLambda import MiCommaLambda
from Code.MiPlusLambda import MiPlusLambda
from Code.OnePlusOneMean import OnePlusOneMean
from Code.OnePlusOneRandom import OnePlusOneRandom
from Code.Pymoo import Pymoo
from Code.utils.data_information import get_structure_data_statistics, get_change_data_statistics
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

means_structure = get_structure_data_statistics(percentage_structure_data, isolation_forest)
# get_data_statistics(filtered_changes_data)

structure_change = StructureChange(filtered_changes_data)
structure_change_model = structure_change.get_model()

evolution_platform = EvolutionPlatform(isolation_forest, structure_change_model)

number_of_companies = 10
evolution_platform.load_companies(number_of_companies)
evolution_platform.generate_start_companies(number_of_companies, means_structure)

evolution_platform.show_structures()


# mean_changes, std_changes = get_change_data_statistics(filtered_changes_data)
#
# one_plus_one_random = OnePlusOneRandom(evolution_platform)
# one_plus_one_mean = OnePlusOneMean(evolution_platform, mean_changes, std_changes)
# mi = 10
# la = 5
# factor = 8
# mi_plus_lambda = MiPlusLambda(evolution_platform, mi, la, factor)
# mi_comma_lambda = MiCommaLambda(evolution_platform, mi, la, factor)
#
# # evolution_platform.add_evolution_strategy(one_plus_one_random)
# # evolution_platform.add_evolution_strategy(one_plus_one_mean)
# evolution_platform.add_evolution_strategy(mi_plus_lambda)
# # evolution_platform.add_evolution_strategy(mi_comma_lambda)
#
# epochs = 10
# evolution_platform.start_evolution(epochs)
#
# evolution_platform.show_all()


def run_company_evolution(companies, ml_model, n_generations=20):
    for generation in range(n_generations):
        print(f"\n--- Generation {generation + 1} ---")
        problem = Pymoo(companies, ml_model)

        algorithm = ES(pop_size=10, offspring_size=10, sigma=0.5)  # smaller sigma → smaller mutations
        termination = get_termination("n_gen", 1)  # one generation at a time

        minimize(problem,
                          algorithm,
                          termination,
                          seed=42,
                          verbose=False)

        problem.apply_best_changes()


run_company_evolution(evolution_platform.generated_companies, structure_change_model, 10)

print("\n=== Final Companies ===")
sorted_companies = sorted(evolution_platform.generated_companies, key=lambda c: -c.value)
for i, company in enumerate(sorted_companies):
    print(f"Company {i + 1}: Value = {company.value:.2f}, Structure = {company.to_array()}")