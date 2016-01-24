# This code provide tools for preparation and creation of classifiers:
# 1. Cleaning values extrcted directly from pandas data frame
# 2. RandomForestClassifier creation using RandomizedSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib


def clean_data_values(data):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    imp_data = imp.transform(data)

    return imp_data


def model_rfc(data, target, estimators_nbr, n_iter_search, joblib_dump_name):
    # build a classifier
    clf = RandomForestClassifier(n_estimators=estimators_nbr)

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search,
                                       scoring='recall_weighted')
    random_search.fit(data, target)

    joblib.dump(clf, joblib_dump_name)
    return random_search
