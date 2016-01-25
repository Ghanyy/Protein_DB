# 1. Load all_summary.txt from http://www.cs.put.poznan.pl/dbrzezinski/teaching/zed/zed_projekt_2015-2016_dane.7z
# 2. Delete rows with res_name equal to: “DA”,“DC”,“DT”, “DU”, “DG”, “DI”,“UNK”, “UNX”, “UNL”, “PR”, “PD”, “Y1”, “EU”,
# “N”, “15P”, “UQ”, “PX4” or “NAN”.
# 3. Filter only unique pairs of (pdb_code, res_name).
# 4. Discard res_name groups with count lesser than 5.
# 5. Load grouped_res_name.txt which is two column file (11005 rows) containing (ln, grouped res_name's).
# 6. Load testing data frame and extract column names in order to extract data for learning process.
# 7. Use RandomForestClassifier to teach simple res_name based classifier
# 8. Use RandomForestClassifier to teach grouped res_name_group based classifier
# 9. Make prediction for both classifiers
# 10. Save both predictions vectors
import preparation as prep
import learning as lrn
import operator
import pickle

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
    data_frame_2_target = data_frame_2.loc[:, "res_name_group"].tolist()

    data_frame_training_data = lrn.clean_data_values(data_frame_training_data)
    # verbose output: UserWarning: Deleting features without observed values:
    # [690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707
    #  708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725
    #  726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743
    #  744 745 746 747 748 749 750 751 752 753 754 755 756 757 758]
    data_columns_exclude = [690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
                            708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725,
                            726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743,
                            744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 0]

    simple_model = lrn.model_rfc(data_frame_training_data, data_frame_target, 50, 50, "data/tmp/simple_model.pkl")
    grouped_model = lrn.model_rfc(data_frame_training_data, data_frame_2_target, 50, 50, "data/tmp/grouped_model.pkl")

    test_df.drop(test_df.columns[data_columns_exclude], axis=1, inplace=True)
    test_data = test_df.values
    test_data = lrn.clean_data_values(test_data)

    simple_prediction = lrn.predict(simple_model, test_data)
    grouped_prediction = lrn.predict(grouped_model, test_data)

    pickle.dump(simple_prediction, open("data/tmp/simple_prediction.p", "wb"))
    pickle.dump(grouped_prediction, open("data/tmp/grouped_prediction.p", "wb"))
