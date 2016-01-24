from sklearn import tree
from sklearn.preprocessing import Imputer

def modelDTC(data, target):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(data)
    imp_data = imp.transform(data)

    model = tree.DecisionTreeClassifier()
    model.fit(imp_data, target)

    return model