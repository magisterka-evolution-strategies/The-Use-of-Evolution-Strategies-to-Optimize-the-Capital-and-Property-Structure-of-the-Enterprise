import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

from Code.utils.data_visualization import visualize_all


class Outliers:
    def __init__(self, percentage_structure_data):
        self.df_outliers = pd.DataFrame(percentage_structure_data,
                                        columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets",
                                                 "CurrentAssets",
                                                 "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital",
                                                 "OwnShares",
                                                 "EquityShareholdersOfTheParent", "NonControllingInterests",
                                                 "NonCurrentLiabilities",
                                                 "CurrentLiabilities",
                                                 "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

        self.df_outliers = self.df_outliers.drop(["CompanyID", "Period", "MarketValue"], axis=1)

    def train_model(self, test_size=0.1, contamination=0.01):
        X_outliers_train, X_outliers_test = train_test_split(self.df_outliers, test_size=test_size)

        self.isolation_forest = IsolationForest(contamination=contamination)
        self.isolation_forest.fit(X_outliers_train)
        test_outliers = self.isolation_forest.predict(X_outliers_test)

        test_outliers = X_outliers_test[test_outliers == -1]

        print("Number of outliers in test data:", len(test_outliers), "out of:", len(X_outliers_test))

        # dbscan = DBSCAN(eps=10, min_samples=2)
        # dbscan.fit(X_train)
        # test_labels = dbscan.fit_predict(X_test)
        #
        # test_outliers = X_test[test_labels == -1]
        #
        # print("Number of outliers in test data:", len(test_outliers))

        return self.isolation_forest

    def check(self, percentage_structure_data):
        X = pd.DataFrame(percentage_structure_data,
                         columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets",
                                  "CurrentAssets",
                                  "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital",
                                  "OwnShares",
                                  "EquityShareholdersOfTheParent", "NonControllingInterests",
                                  "NonCurrentLiabilities",
                                  "CurrentLiabilities",
                                  "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

        X_features = X.drop(["CompanyID", "Period", "MarketValue"], axis=1)

        predictions = self.isolation_forest.predict(X_features)

        valid_structures = [structure for structure, pred in zip(percentage_structure_data, predictions) if pred == 1]

        print("Number of outliers in test data:", sum(predictions == -1), "out of:", len(X))

        only_structure = np.array(np.stack(percentage_structure_data)[:, 3:], dtype=float)
        only_structure_filtered = np.array(np.stack(valid_structures)[:, 3:], dtype=float)

        pca = PCA(n_components=2, random_state=42)
        pca.fit_transform(only_structure)
        visualize_all(only_structure, only_structure_filtered, 'Anomalie wśród struktur kapitałowych', '#ff0000', pca, [-120, 80], [-100, 100])

    def get_mean_values(self):
        return self.df_outliers.mean()
