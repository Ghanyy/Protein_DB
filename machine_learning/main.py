# 1. Load all_summary.txt from http://www.cs.put.poznan.pl/dbrzezinski/teaching/zed/zed_projekt_2015-2016_dane.7z
# 2. Delete rows with res_name equal to: “DA”,“DC”,“DT”, “DU”, “DG”, “DI”,“UNK”, “UNX”, “UNL”, “PR”, “PD”, “Y1”, “EU”,
# “N”, “15P”, “UQ”, “PX4” or “NAN”.
# 3. Filter only unique pairs of (pdb_code, res_name)
# 4. Discard res_name groups with count lesser than 5.
# 5. Load grouped_res_name.txt which is one column file of length 11005 rows containing grouped res_name per row
# concatenated with '_'
# 6. Load testing data frame and extract column names in order to extract data for learning process
# 7. Use DecissionTreeClassifier to teach simple res_name based classifier


import preparation as prep
import learning as lrn
import operator

if __name__ == "__main__":
    data_frame = prep.load_data_frame("data/all_summary.txt", ";", "nan")
    data_frame = prep.filter_unwated_rows(data_frame, ["res_name"], [["DA", "DC", "DT", "DU", "DG", "DI", "UNK", "UNX",
                                                                      "UNL", "PR", "PD", "Y1", "EU", "N", "15P", "UQ",
                                                                      "PX4", "NAN"]])
    data_frame = prep.filter_unique_groups(data_frame, [["pdb_code", "res_name"]])
    data_frame = prep.filter_count_treshold(data_frame, ["res_name"], [operator.ge], [5])
    # len(data_frame) = 11005

    data_frame_2 = prep.load_data_frame("data/grouped_res_name.txt", ",", "")
    # len(data_frame_2) = 11005

    test_df = prep.load_data_frame("data/test_data.txt", ",", "nan")
    # len(test_df) = 18917

    # first parameter is empty (in fact it's an id), ignore it
    data_columns = list(test_df.columns.values)[1:]

    data_frame_training_data = data_frame.loc[:, data_columns].values
    data_frame_target = data_frame.loc[:, "res_name"].tolist()

    simple_model = lrn.modelDTC(data_frame_training_data, data_frame_target)