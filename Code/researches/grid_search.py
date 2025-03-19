import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

from Code.utils.data_modification import get_percentage_structure, get_structure_changes, get_filtered_changes
from Code.utils.retrieve_data import get_raw_sql_data


filename = "../exporter_new.db"

data = get_raw_sql_data(filename)

# company_id, period, market_value, 5 x assets, 5 x liabilities
percentage_structure_data = get_percentage_structure(data)

structure_changes_data = get_structure_changes(percentage_structure_data)

filtered_changes_data = get_filtered_changes(structure_changes_data)
print(filtered_changes_data[0])

df = pd.DataFrame(filtered_changes_data, columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets", "CurrentAssets",
                                       "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                       "EquityShareholdersOfTheParent", "NonControllingInterests",
                                       "NonCurrentLiabilities",
                                       "CurrentLiabilities",
                                       "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

print(df)

df = df.drop(["CompanyID", "Period"], axis=1)
X = df.drop(["MarketValue"], axis=1)
y = df["MarketValue"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.07)


def create_model(n_nodes):
    inputs = keras.Input(shape=(X.shape[1],))
    model = keras.models.Sequential()
    model.add(inputs)
    model.add(keras.layers.Dense(n_nodes[0], activation="relu"))
    for i in range(1, len(n_nodes)):
        model.add(keras.layers.Dense(n_nodes[i], activation="relu"))
        model.add(keras.layers.Dropout(0.1))
    outputs = keras.layers.Dense(1)
    model.add(outputs)

    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model


# Define the hyperparameters grid
param_grid = {
    'n_nodes': [[64, 32], [64, 64], [128, 64], [128, 128], [64, 32, 16], [16, 8, 4, 2], [64, 16, 4], [128, 64, 32, 16],
                [128, 64, 32, 16, 8, 4, 2]],
}

keras_regressor = KerasRegressor(model=create_model, epochs=100, n_nodes=[64, 32])

grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, cv=5)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_model = grid_result.best_estimator_.model
print(best_model)
