from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn import datasets
import skfuzzy.control as ctrl
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


filename = 'weatherAUS.csv'
dataset = read_csv(filename, low_memory=False)

df = dataset
# print(df.info())

features_list = list()
accuracy_list = list()
rules_list = list()

for i in range(0,100):
    df_temp = pd.DataFrame()
    randomlist = random.sample(range(0, 22), 5)
    col_list = list(range(0, 23))
    # print(randomlist)
    # print(col_list)
    drop_col = list(set(col_list) - set(randomlist)) + list(set(randomlist) - set(col_list))
    # print(drop_col)
    
    df_temp = df.drop(df.columns[drop_col], axis=1)
    # print(df_temp.head(10))
    # print(df_temp.columns)
    print("\n")
    
    df_temp = df_temp.replace(['NA'], np.nan)
    df_num = df_temp.select_dtypes(include=[np.number])
    df_cat = df_temp.select_dtypes(exclude=[np.number])
    df_cat["RainTomorrow"] = dataset["RainTomorrow"]
    # print(df_cat.head(10))

    num_col = df_num.columns
    num_col_len = df_num.shape[1]
    for j in range(0,num_col_len):
            df_num.iloc[:, j]=df_num.iloc[:, j].fillna(df_num.iloc[:, j].mean())
    
    cat_col_len = df_cat.shape[1]
    for j in range(0,cat_col_len):
            df_cat.iloc[:, j]=df_cat.iloc[:, j].fillna(df_cat.iloc[:, j].mode()[0])

    le = LabelEncoder()

    for j in range(0,cat_col_len):
            df_cat.iloc[:, j]=le.fit_transform(df_cat.iloc[:, j])

    trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
    data = trans.fit_transform(df_num)
    df_num = DataFrame(data)

    des_df = df_num.join(df_cat)

    for j in range(0,num_col_len):
        des_df[num_col[j]] = df_num.iloc[:, j]
        
    drop_list = list(range(0, num_col_len))
    des_df_new = des_df.drop(des_df.columns[drop_list], axis=1)

    des_df_new_col = des_df_new.columns

    temp = ['Temp3pm','Temp9am','Humidity9am',"Date"]
    col_drop = [value for value in des_df_new_col if value in temp]

    # print(des_df_new_col)
    if(len(col_drop) > 0):
        for j in range(0, len(col_drop)):
              des_df_new.drop(col_drop[j], axis=1, inplace = True)

    # des_df_new["RainTomorrow"] = dataset["RainTomorrow"]
    des_df_new_col = des_df_new.columns
    last_col = des_df_new.pop("RainTomorrow")
    des_df_new.insert((len(des_df_new_col)-1), "RainTomorrow", last_col)
    # print(des_df_new.head(10))
    des_df_new_col = des_df_new.columns
    
    os = SMOTE()
    x, y = os.fit_resample(des_df_new.iloc[:,:-1], des_df_new.iloc[:,-1])
    count = Counter(y)
    print(count)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    report3 = classification_report(y_test, y_pred)
    # print(report3)
    accuracy = accuracy_score(y_test,y_pred)
    print("Accuracy of the Decision Tree Model is:",accuracy)
    cm3 = confusion_matrix(y_test, y_pred)
    # print(cm3)
    print(des_df_new.head(10))

    text_repr = get_rules(clf, feature_names = list(X_train.columns), class_names = [0,1])

    rules_list.append(text_repr)

    features_list.append(des_df_new_col)

    accuracy_list.append(accuracy)
       
    del df_temp
    del df_num
    del df_cat
    del des_df
    del des_df_new


max_acc = accuracy_list.index(max(accuracy_list))
print('Best Accuracy:' + str(accuracy_list[max_acc]))
# features = features_list[max_acc] 
# print(features_list[max_acc])

# text_repr = rules_list[max_acc]

# filename = 'discretized.csv'
# final_data = to_csv(filename, low_memory=False)

