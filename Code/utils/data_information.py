import pandas as pd


def get_data_statistics(data, outliers_model):
    df = pd.DataFrame(data,
                      columns=["CompanyID", "Period", "MarketValue", "NonCurrentAssets", "CurrentAssets",
                               "AssetsHeldForSaleAndDiscountinuingOperations", "CalledUpCapital", "OwnShares",
                               "EquityShareholdersOfTheParent", "NonControllingInterests",
                               "NonCurrentLiabilities",
                               "CurrentLiabilities",
                               "LiabilitiesRelatedToAssetsHeldForSaleAndDiscontinuedOperations"])

    df = df.drop(["CompanyID", "Period", "MarketValue"], axis=1)

    predictions = outliers_model.predict(df)

    standard_structures = df[predictions == 1]

    r1 = standard_structures.mean()
    # print("\nMean:")
    # print(r1)

    r2 = standard_structures.std()
    # print("\nstd:")
    # print(r2)

    min_values = standard_structures.min()
    # print("\nMin:")
    # print(min_values)

    max_values = standard_structures.max()
    # print("\nMax:")
    # print(max_values)

    return r1