from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn import datasets
# import skfuzzy.control as ctrl
import random
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,roc_curve, roc_auc_score
from collections import Counter
from sklearn import tree
from sklearn.tree import _tree


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules



filename = 'discretized.csv'
dataset = read_csv(filename, low_memory=False)

os = SMOTE()
x, y = os.fit_resample(dataset.iloc[:,:-1], dataset.iloc[:,-1])
count = Counter(y)
print(count)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

text_repr = get_rules(clf, feature_names = list(X_train.columns), class_names = [0,1])

with open('final_rules.txt', 'w+') as f:
    for items in text_repr:
        f.write('%s\n\n' %items)
    print("File written successfully")
f.close()
