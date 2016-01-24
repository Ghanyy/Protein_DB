# 1. Load all_summary.txt from http://www.cs.put.poznan.pl/dbrzezinski/teaching/zed/zed_projekt_2015-2016_dane.7z
# 2. Delete rows with res_name equal to: “DA”,“DC”,“DT”, “DU”, “DG”, “DI”,“UNK”, “UNX”, “UNL”, “PR”, “PD”, “Y1”, “EU”,
# “N”, “15P”, “UQ”, “PX4” or “NAN”.
# 3. Filter only unique pairs of (pdb_code, res_name)
# 4. Discard res_name groups with count lesser than 5.

import preparation as prep
import operator

if __name__ == "__main__":
    data_frame = prep.load_data_frame("data/all_summary.txt", ";", "nan")
    data_frame = prep.filter_unwated_rows(data_frame, ["res_name"], [["DA", "DC", "DT", "DU", "DG", "DI", "UNK", "UNX",
                                                                      "UNL", "PR", "PD", "Y1", "EU", "N", "15P", "UQ",
                                                                      "PX4", "NAN"]])
    data_frame = prep.filter_unique_groups(data_frame, [["pdb_code", "res_name"]])
    data_frame = prep.filter_count_treshold(data_frame, ["res_name"], [operator.ge], [5])