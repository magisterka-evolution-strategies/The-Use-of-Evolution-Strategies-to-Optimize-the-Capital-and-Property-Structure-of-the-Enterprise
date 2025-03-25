import os

import keras
import pandas as pd
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StructureChange:
    def __init__(self, filtered_changes_data):
        df = pd.DataFrame(filtered_changes_data,
                          columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets", "CurrentAssets",
                                   "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                                   "EquityShareholdersOfTheParent", "NonControllingInterests",
                                   "NonCurrentLiabilities",
                                   "CurrentLiabilities",
                                   "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

        df = df.drop(["CompanyID", "Period"], axis=1)
        self.X = df.drop(["MarketValue"], axis=1)
        self.y = df["MarketValue"]

        self.model_path = "models/structure_change_model.h5"

    def get_model(self):
        if os.path.exists(self.model_path):
            print("Model found! Loading existing model...")
            model = load_model(self.model_path)
        else:
            print("No model found. Training a new model...")
            model = self.train_model()

        self.model = model
        return self.model

    def train_model(self):
        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()
        # X_scaled = scaler_X.fit_transform(self.X)
        # y_scaled = scaler_y.fit_transform(self.y.values.reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.15)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.07)

        n_nodes = [128, 64, 32, 16]

        inputs = keras.Input(shape=(self.X.shape[1],))
        model = keras.models.Sequential()
        model.add(inputs)
        model.add(keras.layers.Dense(n_nodes[0], activation="relu"))
        for i in range(1, len(n_nodes)):
            model.add(keras.layers.Dropout(0.1))
            model.add(keras.layers.Dense(n_nodes[i], activation="relu"))
        outputs = keras.layers.Dense(1)
        model.add(outputs)

        optimizer = keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_absolute_error"])

        model.fit(
            X_train,
            y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            validation_freq=10,
            validation_split=0.2,
            # callbacks=[
            #     keras.callbacks.EarlyStopping(monitor="mean_absolute_error", patience=10),
            # ],
            verbose=True
        )

        predictions = model.predict(X_test)
        for i in range(len(predictions)):
            print(predictions[i], y_test.iloc[i])

        loss = model.evaluate(X_test, y_test)
        print("Test Loss:", loss)

        model.save(self.model_path)

        return model
