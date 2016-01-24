# This code provide tools for initial processing of PDB (Protein Data Bank) data set:
# 1. Opening targeted file.
# 2. Filtering unwanted rows in selected data set column.
# 3. Filtering only unique values of group columns.
# 4. Filtering rows with count threshold.

import pandas as pd


def load_data_frame(file_path, column_separator, nan_text):
    if file_path != '' and column_separator != '':
        if nan_text != '':
            df = pd.read_csv(file_path, sep=column_separator, na_values=nan_text, keep_default_na=False, dtype='unicode')
        else:
            df = pd.read_csv(file_path, sep=column_separator, dtype='unicode')

        return df
    else:
        raise Exception('Mandatory parameters missing')


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


def filter_unique_groups(df, groups):
    if df is not None and len(groups) > 0:
        filtered_df = df
        for group in groups:
            filtered_df = filtered_df.drop_duplicates(subset=group)

        return filtered_df
    else:
        raise Exception('Mandatory parameters missing')


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