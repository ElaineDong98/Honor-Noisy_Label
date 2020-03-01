import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

import truthFinder

df = pd.read_csv("/data/mimic/generated_data/features.csv")
x = df
#list(df.columns)
df_result = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.469_variance_0.001_10_nurses.csv')
df_result2 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.564_variance_0.001_10_nurses.csv')
df_result3 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.565_variance_0.007_10_nurses.csv')
df_result4 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.714_variance_0.001_10_nurses.csv')
df_result5 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.714_variance_0.003_10_nurses.csv')
df_result6 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.781_variance_0.010_10_nurses.csv')
df_result7 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.813_variance_0.003_10_nurses.csv')
df_result8 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.824_variance_0.005_10_nurses.csv')
df_result9 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.830_variance_0.001_10_nurses.csv')
df_result10 = pd.read_csv(
    '/data/mimic/generated_data/annotations_mean_reliability_0.848_variance_0.001_10_nurses.csv')
df_result_set = [df_result, df_result2, df_result3, df_result4,
                 df_result5, df_result6, df_result7, df_result8, df_result9, df_result10]
tag = df_result[['HADM_ID', 'ground_truth']]

#list(df_result.columns)


def majoriy(multi_labels):
    pred = []
    for i in range(len(multi_labels[0])):
        compare = 0
        for j in range(10):
            temp = -1 if multi_labels[j][i] == 0 else 1
            compare += temp
        if (compare < 0):
            pred.append(0)
        else:
            pred.append(1)
    return pred


def accuracy_ratio(pred, ans):
    count = 0
    l = len(pred)
    for i in range(l):
        if(pred[i] == ans[i]):
            count += 1
    return 1.0*count/l


def weight(multi_labels):
    # weight
    w = [1.0]*10
    #count
    c = [0]*10
    pred = []
    result = -1
    for i in range(len(multi_labels[0])):
        compare = 0
        for j in range(10):
            # change it to 1, -1
            temp = -1 if multi_labels[j][i] == 0 else 1
            compare += w[j] * temp
        result = 1 if compare > 0 else 0
        pred.append(result)
        for j in range(10):
            if multi_labels[j][i] == result:
                c[j] += 1
            else:
                w[j] *= 0.5
        stand = 1.0*sum(c) / len(c)
#         for j in range(10):
#             w[j] = 1.0 * c[j] / (i+1)
#         for j in range(10):
#             if c[j] > stand:
#                 w[j] *= 1.2
#             elif c[j] < stand:
#                 w[j] *= 0.8

    #print("weight: ",w)
    return pred


def generic_clf(Y_train, X_train, Y_test, X_test, clf, y_true_train, y_true_test):
	clf.fit(X_train, Y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	return get_error_rate(pred_train, Y_train), \
		get_error_rate(pred_test, Y_test), \
		get_error_rate(pred_train, y_true_train), \
		get_error_rate(pred_test, y_true_test)
#adaboost


def get_error_rate(pred, Y):
	return (sum(pred != Y) / float(len(Y)))


indices = range(len(df.index))
X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
    x, tag['ground_truth'], indices, test_size=0.2, random_state=15)
X_train, X_val, y_train, y_val, ind_train, ind_val = train_test_split(
    X_train, y_train, ind_train, test_size=0.25, random_state=15)


#list_experts = df_result[['target_0', 'target_1','target_2','target_3','target_4','target_5','target_6','target_7','target_8','target_9']]
list_experts = []
list_sum = []
for i in range(10):
    df_result_set[i].values.T.tolist()
    list_expert = [list(l) for l in zip(*df_result_set[i].values)]
    list_expert = list_expert[:10]
    list_experts.append(list_expert)
    list_sum.append([list(a) for a in zip(*list_experts[i])])

##########################
y_trains_wei = []
y_tests_wei = []
y_vals_wei = []
for z in range(10):
    print("-------------For dataset "+str(z+1)+"---------")
    i = 0
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = 0
    for each in range(len(x)):
        for i in range(10):
            if list_experts[z][i][index] == tags[z]['ground_truth'][index]:
                result[i] += 1
        index += 1
    print("accuracy: ", [1.0*accuracy / len(tags[z]
                                            ['ground_truth']) for accuracy in result])
    major_pred = majoriy(list_experts[z])
    weight_pred = weight(list_experts[z])

    claims = dict()
    claims['source_id'] = []
    claims['object_id'] = []
    claims['value'] = []
    for i in range(10):
        for j in range(len(x)):
            claims['source_id'].append(i)
            claims['object_id'].append(j)
            claims['value'].append(list_sum[z][j][i])
    claims = pd.DataFrame(data=claims)
    trust_df, truth_df = truthFinder.truthfinder(claims,
                                                 imp_func=truthFinder.imp,
                                                 initial_trust=0.99,
                                                 similarity_threshold=(
                                                     1 - 1e-05),
                                                 dampening_factor=0.3,
                                                 verbose=False)
    find_pred = []
    find_pred.append(truth_df["value"][0])
    conf = truth_df["confidence"][0]
    id = truth_df["object_id"][0]
    for i in range(1, len(truth_df)):
        if truth_df["object_id"][i] == id:
            if (truth_df["confidence"][i] > conf):
                find_pred[id] = truth_df["value"][i]
        else:
            find_pred.append(truth_df["value"][i])
        conf = truth_df["confidence"][i]
        id = truth_df["object_id"][i]

    print("accuracy of majority vote pred: ", accuracy_ratio(major_pred, tags[z]['ground_truth']),
          "\naccuracy of weighted: ", accuracy_ratio(
              weight_pred, tags[z]['ground_truth']),
          "\naccuracy of truthFinder: ", accuracy_ratio(find_pred, tags[z]['ground_truth']))

    X_train_wei, X_test_wei, y_train_wei, y_test_wei = train_test_split(
        x, weight_pred, test_size=0.2, random_state=15)
    X_train_wei, X_val_wei, y_train_wei, y_val_wei = train_test_split(
        X_train_wei, y_train_wei, test_size=0.25, random_state=15)

    y_trains_wei.append(y_train_wei)
    y_tests_wei.append(y_test_wei)
    y_vals_wei.append(y_val_wei)

    Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = pd.DataFrame(Imputer.fit_transform(X_train_wei))
    X_train.columns = X_train_wei.columns

    X_val = pd.DataFrame(Imputer.fit_transform(X_val_wei))
    X_val.columns = X_val_wei.columns

    X_test = pd.DataFrame(Imputer.fit_transform(X_test_wei))
    X_test.columns = X_val_wei.columns

    clf_tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf.fit(X_train, y_train_wei)
    er_tree_wei = generic_clf(y_train_wei, X_train,
                              y_val_wei, X_val, clf_tree, y_train, y_val)

    baseline_tree = generic_clf(y_train_wei, X_train,
                                y_test_wei, X_test, clf_tree, y_train, y_test)
    baseline_tree_t = generic_clf(y_train, X_train,
                                  y_test, X_test, clf_tree, y_train, y_test)
    baseline_rd = clf_rf.predict(X_test)
    print("----baseline ----\nthe accurary is:",
          str(1-baseline_tree[3]), " for DT\n", str(1-get_error_rate(baseline_rd, y_test)), "for RF")

    clf_rf.fit(X_train, y_train)
    baseline_rd_t = clf_rf.predict(X_test)
    if z == 0:
        print("----baseline when given accurate tag----\nthe accurary is:",
              str(1-baseline_tree_t[3]), " for DT\n", str(1-get_error_rate(baseline_rd_t, y_test)), "for RF")

    print("------------- This Dataset Ends Here---------")

norms = []
for z in range(10):
    norm = []
    w = []
    index_w = 0
    for diag in list_sum[z]:
        temp = 0
        for each in diag:
            temp = (temp + 1) if each == 1 else (temp - 1)
        if index_w in ind_train:
            w.append(1.0/(1+(10**abs(temp))))
        index_w += 1
    norm = [1.0*float(i)/sum(w) for i in w]
    norms.append(norm)

for z in range(10):
    print("-------------For dataset "+str(z+1)+"---------")
    er_train_wei, er_val_wei, er_true_train_wei, er_true_val_wei = [
        er_tree_wei[0]], [er_tree_wei[1]], [er_tree_wei[2]], [er_tree_wei[3]]
    x_range = range(10, 450, 20)
    low_error = 1
    best_iter = 0
    for i in x_range:
        clf = AdaBoostClassifier(n_estimators=i, random_state=15)
        clf.fit(X_train, y_trains_wei[z], sample_weight=norms[z])
        pred_i_wei_val = clf.predict(X_val)
        pred_i_wei_train = clf.predict(X_train)
        er_train_wei.append(get_error_rate(pred_i_wei_train, y_trains_wei[z]))
        er_val_wei.append(get_error_rate(pred_i_wei_val, y_vals_wei[z]))
        er_true_train_wei.append(get_error_rate(pred_i_wei_train, y_trains[z]))
        er_true_val_wei.append(get_error_rate(pred_i_wei_val, y_vals[z]))
        if(low_error > er_val_wei[-1]):  # this is compared to our y
            low_error = er_val_wei[-1]
            best_iter = i
    clf = AdaBoostClassifier(n_estimators=best_iter, random_state=15)
    clf.fit(X_train, y_trains_wei[z], sample_weight=norms[z])
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    er_true_test_wei = get_error_rate(pred_test, y_test)
    er_test_wei = get_error_rate(pred_test, y_tests_wei[z])

    er_train_wei_l, er_val_wei_l = [er_tree_wei[0]], [er_tree_wei[1]]
    er_true_train_wei_l, er_true_val_wei_l = [er_tree_wei[2]], [er_tree_wei[3]]
    learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    low_error = 1
    best_lr = 0
    best_lr_index = 0
    index = -1
    for eta in learning_rates:
        index += 1
        clf = AdaBoostClassifier(
            n_estimators=best_iter, random_state=15, learning_rate=eta)
        clf.fit(X_train, y_train_wei, sample_weight=norms[z])
        pred_i_wei_val = clf.predict(X_val)
        pred_i_wei_train = clf.predict(X_train)
        er_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train_wei))
        er_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val_wei))
        er_true_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train))
        er_true_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val))
        if(low_error > er_val_wei_l[-1]):  # this is compared to our y
            low_error = er_val_wei_l[-1]
            best_lr = eta
            best_lr_index = index
    clf = AdaBoostClassifier(n_estimators=best_iter,
                             random_state=15, learning_rate=best_lr)
    clf.fit(X_train, y_train_wei, sample_weight=norms[z])
    y_score = clf.decision_function(X_test)

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    er_true_test_wei = get_error_rate(pred_test, y_test)
    er_test_wei = get_error_rate(pred_test, y_test_wei)
    print("Using the best (iteration, learning rate)= (" + str(best_iter) +
          ", " + str(best_lr) + "),\nthe accuracy is: ", 1.0 - er_true_test_wei)

    if z == 0:
        clf = AdaBoostClassifier(
            n_estimators=best_iter, random_state=15, learning_rate=best_lr)
        clf.fit(X_train, y_train_wei)
        y_score = clf.decision_function(X_test)

        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
        er_true_test_wei = get_error_rate(pred_test, y_test)
        er_test_wei = get_error_rate(pred_test, y_test_wei)
        print("Originally: Using the best (iteration, learning rate)= (" + str(best_iter) +
              ", " + str(best_lr) + "),\nthe error rate is: ", 1.0 - er_true_test_wei)

    print("------------- This Dataset Ends Here---------")

for z in range(10):
    print("-------------For dataset "+str(z+1)+"---------")
    er_train_wei_l, er_val_wei_l, er_true_train_wei_l, er_true_val_wei_l = [], [], [], []

    learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
    low_error = 1
    best_lr = 0
    best_lr_index = 0
    index = -1
    for eta in learning_rates:
        index += 1
        gb = GradientBoostingClassifier(random_state=15, learning_rate=eta)
        gb.fit(X_train, y_trains_wei[z], sample_weight=norms[z])
        pred_i_wei_val = gb.predict(X_val)
        pred_i_wei_train = gb.predict(X_train)
        er_train_wei_l.append(get_error_rate(
            pred_i_wei_train, y_trains_wei[z]))
        er_val_wei_l.append(get_error_rate(pred_i_wei_val, y_vals_wei[z]))
        er_true_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train))
        er_true_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val))
        if(low_error > er_val_wei_l[-1]):  # this is compared to our y
            low_error = er_val_wei_l[-1]
            best_lr = eta
            best_lr_index = index

    er_train_wei_l, er_val_wei_l, er_true_train_wei_l, er_true_val_wei_l = [], [], [], []
    subsampling = [0.6, 0.7, 0.8, 0.9, 1.0]
    low_error = 1
    best_sub = 0
    best_sub_index = 0
    index = -1
    for sub_s in subsampling:
        index += 1
        gb = GradientBoostingClassifier(
            random_state=15, learning_rate=0.1, subsample=sub_s)
        gb.fit(X_train, y_trains_wei[z], sample_weight=norms[z])
        pred_i_wei_val = gb.predict(X_val)
        pred_i_wei_train = gb.predict(X_train)

        er_train_wei_l.append(get_error_rate(
            pred_i_wei_train, y_trains_wei[z]))
        er_val_wei_l.append(get_error_rate(pred_i_wei_val, y_vals_wei[z]))
        er_true_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train))
        er_true_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val))
        if(low_error > er_val_wei_l[-1]):  # this is compared to our y
            low_error = er_val_wei_l[-1]
            best_sub = sub_s
            best_sub_index = index
    clf = GradientBoostingClassifier(
        random_state=15, learning_rate=best_lr, subsample=best_sub)
    clf.fit(X_train, y_trains_wei[z], sample_weight=norms[z])
    pred_train = clf.predict(X_train)
    pred_test_g = clf.predict(X_test)
    er_true_test_wei = get_error_rate(pred_test_g, y_test)
    er_test_wei = get_error_rate(pred_test_g, y_tests_wei[z])
    print("Using the best subsample = " + str(best_sub) +
          "\nthe accuracy is: ", 1.0 - er_true_test_wei)

    clf = GradientBoostingClassifier(
        n_estimators=100, random_state=15, learning_rate=best_lr, subsample=best_sub)
    clf.fit(X_train, y_trains_wei[z])
    pred_train = clf.predict(X_train)
    pred_test_g = clf.predict(X_test)
    er_true_test_wei = get_error_rate(pred_test_g, y_test)
    er_test_wei = get_error_rate(pred_test_g, y_tests_wei[z])
    print("Originally: Using the best subsample = " + str(best_sub) +
          "\nthe accuracy is: ", 1.0 - er_true_test_wei)

    print("------------- This Dataset Ends Here---------")
