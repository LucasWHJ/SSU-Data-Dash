"""Models for the off-market and list-aware home-price tracks."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features


def make_model(track):
    """Return the active model for a competition track."""
    if track == "off_market":
        return FeatureModel(features.off_market_features, learning_rate=0.18,
                            max_iter=400, max_leaf_nodes=60, min_samples_leaf=30,
                            max_features=0.7)
    if track == "list_aware":
        return FeatureModel(features.list_aware_features, relative_to_list=True,
                            loss="absolute_error")
    raise ValueError(f"Unknown track: {track}")


class ListPriceBaseline:
    """List-aware baseline: predict exactly the asking price."""

    def fit(self, train):
        return self

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: median of the nearest earlier sales of the same type."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """Gradient-boosted trees fitted only on the supplied past sales.

    Off-market uses squared error on log(sale price). List-aware uses absolute
    error on log(sale / list price), with the asking price as an anchor.
    Categories come from the training sales and are reused unchanged for
    validation and prediction.
    """

    def __init__(self, make_features, relative_to_list=False, loss="squared_error",
                 learning_rate=0.06, max_iter=260, max_leaf_nodes=31,
                 min_samples_leaf=40, max_features=1.0):
        self.make_features = make_features
        self.relative_to_list = relative_to_list
        self.categories = {}
        self.model = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            l2_regularization=0.1,
            categorical_features="from_dtype",
            early_stopping=False,
            random_state=0,
        )

    def _prepare(self, homes, fitting=False):
        x = self.make_features(homes).copy()
        for column in features.CATEGORICAL_COLUMNS:
            if fitting:
                self.categories[column] = pd.Index(x[column].unique()).tolist()
            x[column] = pd.Categorical(x[column], categories=self.categories[column])
        return x

    def fit(self, train):
        x = self._prepare(train, fitting=True)
        target = train["sold_price"].to_numpy(dtype=float)
        if self.relative_to_list:
            target = target / train["list_price"].to_numpy(dtype=float)
        self.model.fit(x, np.log(target))
        return self

    def predict(self, homes):
        prediction = np.exp(self.model.predict(self._prepare(homes)))
        if self.relative_to_list:
            prediction *= homes["list_price"].to_numpy(dtype=float)
        return prediction
