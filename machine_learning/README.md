
### Protein DB - machine learning
### Ghanyy, 106080
#### 24 stycznia 2016

# Introduction

Purpose of this project is to create simple classification based on [Protein Data Bank](http://www.rcsb.org/) (PDB) data analysed previously using R.

## Libraries


```python
import pandas as pd # reading raw data, preprocessing
import operator # used only for intelligent filtering
import pickle # saving python array to file
from sklearn.grid_search import RandomizedSearchCV # parameters parametrisation using random search
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.preprocessing import Imputer # clearing NaNs from data frame data
from scipy.stats import randint as sp_randint # random numbers generation
from sklearn.externals import joblib # saving best classifier to file
```

## Loading raw csv data
### Following utility functions where implemented


```python
# Opening targeted file.
def load_data_frame(file_path, column_separator, nan_text):
    if file_path != '' and column_separator != '':
        if nan_text != '':
            df = pd.read_csv(file_path, sep=column_separator, na_values=nan_text, keep_default_na=False, dtype='unicode')
        else:
            df = pd.read_csv(file_path, sep=column_separator, dtype='unicode')

        return df
    else:
        raise Exception('Mandatory parameters missing')


# Filtering unwanted rows in selected data set column.
def filter_unwated_rows(df, columns, values):
    if df is not None and len(columns) > 0 and len(values) > 0:
        if len(columns) == len(values):
            filtered_df = df
            filter_pairs = zip(columns, values)
            for filter_pair in filter_pairs:
                filtered_df = filtered_df[~df[filter_pair[0]].isin(filter_pair[1])]

            return filtered_df
        else:
            raise Exception('Parameters mismatch')
    else:
        raise Exception('Mandatory parameters missing')

        
# Filtering only unique values of group columns.
def filter_unique_groups(df, groups):
    if df is not None and len(groups) > 0:
        filtered_df = df
        for group in groups:
            filtered_df = filtered_df.drop_duplicates(subset=group)

        return filtered_df
    else:
        raise Exception('Mandatory parameters missing')


# Filtering rows with count threshold.
# note: columns is string[], operators is operator[], tresholds is int[]
def filter_count_treshold(df, columns, operators, tresholds):
    if df is not None and len(columns) > 0 and len(tresholds) > 0 and len(operators) > 0:
        if len(columns) == len(tresholds) == len(operators):
            filtered_df = df
            filter_pairs = zip(columns, operators, tresholds)
            for filter_pair in filter_pairs:
                filtered_df = filtered_df[filter_pair[1](df[filter_pair[0]].map(df[[filter_pair[0]]].stack().value_counts()), filter_pair[2])]

            return filtered_df
        else:
            raise Exception('Parameters mismatch')
    else:
        raise Exception('Mandatory parameters missing')
```

### Loading PDB data

```python
# Load all_summary.txt from http://www.cs.put.poznan.pl/dbrzezinski/teaching/zed/zed_projekt_2015-2016_dane.7z
data_frame = prep.load_data_frame("data/all_summary.txt", ";", "nan")

# Delete rows with res_name equal to: “DA”,“DC”,“DT”, “DU”, “DG”, “DI”,“UNK”, “UNX”, “UNL”, “PR”, “PD”, “Y1”, “EU”, “N”, “15P”, “UQ”, “PX4” or “NAN”.
data_frame = prep.filter_unwated_rows(data_frame, ["res_name"], [["DA", "DC", "DT", "DU", "DG", "DI", "UNK", "UNX",
                                                                      "UNL", "PR", "PD", "Y1", "EU", "N", "15P", "UQ",
                                                                      "PX4", "NAN"]])

# Filter only unique pairs of (pdb_code, res_name).
data_frame = prep.filter_unique_groups(data_frame, [["pdb_code", "res_name"]])

# Discard res_name groups with count lesser than 5.
data_frame = prep.filter_count_treshold(data_frame, ["res_name"], [operator.ge], [5])
# len(data_frame) = 11005

# Load grouped_res_name.txt which is two column file (11005 rows) containing (ln, grouped res_name's).
data_frame_2 = prep.load_data_frame("data/grouped_res_name.txt", ",", "")

# Load testing data frame and extract column names in order to extract data for learning process.
test_df = prep.load_data_frame("data/test_data.txt", ",", "nan")
# len(test_df) = 18917
```

## Create classifier

### Following utility functions were implemented


```python
# Cleaning values extracted directly from pandas data frame.
def clean_data_values(data):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    imp_data = imp.transform(data)

    return imp_data


# RandomForestClassifier creation using RandomizedSearchCV.
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

    # save best classifier to file
    joblib.dump(clf, joblib_dump_name)
    return random_search


# Make extremely minimalistic prediction for given test set.
def predict(model, test_data):
    results = model.predict(test_data)
    return results
```

### Creation of RDB classifiers
``` python
# first parameter is empty (in fact it's an id), ignore it
data_columns = list(test_df.columns.values)[1:]

# extract only columns mentioned in test data set
data_frame_training_data = data_frame.loc[:, data_columns].values

# extract target data
data_frame_target = data_frame.loc[:, "res_name"].tolist()
data_frame_2_target = data_frame_2.loc[:, "res_name_group"].tolist()

# clean data from NaNs
data_frame_training_data = lrn.clean_data_values(data_frame_training_data)
# verbose output: UserWarning: Deleting features without observed values:
# [690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707
#  708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725
#  726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743
#  744 745 746 747 748 749 750 751 752 753 754 755 756 757 758]

# training and testing data sets must have matching columns, so it will be necessary to cut those off
data_columns_exclude = [690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
                                           708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725,
                                           726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743,
                                           744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 0]

# Use RandomForestClassifier to teach simple res_name based classifier
simple_model = lrn.model_rfc(data_frame_training_data, data_frame_target, 50, 50, "data/tmp/simple_model.pkl")

# Use RandomForestClassifier to teach grouped res_name_group based classifier
grouped_model = lrn.model_rfc(data_frame_training_data, data_frame_2_target, 50, 50, "data/tmp/grouped_model.pkl")

# drop unnecessary columns and clean data
test_df.drop(test_df.columns[data_columns_exclude], axis=1, inplace=True)
test_data = test_df.values
test_data = lrn.clean_data_values(test_data)

# make predictions
simple_prediction = lrn.predict(simple_model, test_data)
grouped_prediction = lrn.predict(grouped_model, test_data)

#save predictions to file
pickle.dump(simple_prediction, open("data/tmp/simple_prediction.p", "wb"))
pickle.dump(grouped_prediction, open("data/tmp/grouped_prediction.p", "wb"))
```

## Saved data
In order to acuire saved data download .zip'ed catalog `data/tmp`.
Predictions can be "unpickled" using `pickle` library.
Classifiers can be loaded as `RandomForestClassifier` object with `sklearn.externals.joblib` library.


```python

```
