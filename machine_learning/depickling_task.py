# Utility script for depickling predictions for readable format
import pickle


if __name__ == "__main__":
    simple_prediction = pickle.load(open("data/tmp/simple_prediction.p", "rb"))
    grouped_prediction = pickle.load(open("data/tmp/grouped_prediction.p", "rb"))

    simple_prediction_file = open("data/tmp/simple_prediction.txt", "wb")
    for s_prediction in simple_prediction:
        simple_prediction_file.write(("%s\n" % s_prediction).encode(encoding='utf_8'))
    simple_prediction_file.close()

    grouped_prediction_file = open("data/tmp/grouped_prediction.txt", "wb")
    for g_prediction in grouped_prediction:
        grouped_prediction_file.write(("%s\n" % g_prediction).encode(encoding='utf_8'))
    grouped_prediction_file.close()