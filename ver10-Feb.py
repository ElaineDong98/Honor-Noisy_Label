import numpy as np
from numpy.linalg import cholesky
import xlsxwriter
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import auc   
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cosine
from sklearn.datasets import make_blobs
from scipy.spatial import distance
import random
import truthFinder 
from itertools import cycle
from sklearn import tree
import pandas as pd

#v6 new changes:  consider the relationship between our labels and the features
#v7 new changes:  initial weight for adaboost changed
#v8 new changes:  used the truthdiscovery (truth finder)
#v9 new changes:  only left "weighted" option. 
# 				  Saved the result for different initial weights
#				  Added a validation dataset to determine number of iteration
# 				  used GradientBoostingClassifier from sklearn
#v10 new changes: 10 experts
#				  10 features
#				  changed the weighted method



# #for future improvement
# 1. change the weight and plot 
# 2. read paper for gradient tree boosting
# 4. snorkle to find labeling. Crowd sourcing 
#  covarience the same 
#  add more features 
#  add noisy labels
# 5. Linear Discriminant Analysiss
# 6. Try Logistic, De Tree, Random Forest, SVM
# 7. Add graph: Error, Presision, Recall, AUPRC
# 8. Compare pla
# 9. the best iteration rate is the earlist lowest rate. 
# 10. Stochastic 

#majority input: decisions from all experts
#         output: prediction. 
def majoriy(multi_labels):
	pred = []
	for i in range(len(multi_labels[0])):
		compare = 0
		for j in range(10):
			compare += multi_labels[j][i]
		if (compare < 0):
			pred.append(-1)
		else:
			pred.append(1)
	return pred


def accuracy(pred, ans):
	count = 0
	l = len(pred)
	for i in range(l):
		if(pred[i] == ans[i]):
			count += 1 
	return count/l

def weight(multi_labels):
	# weight
	w = [1]*len(multi_labels)
	#count 
	c = [0]*len(multi_labels)
	pred = []
	result = -1
	for i in range(len(multi_labels[0])):
		compare = 0
		for j in range(10):
			compare += w[j] * multi_labels[j][i]
		result = 1 if compare > 0 else -1
		pred.append(result)
		for j in range(10):
			if multi_labels[j][i] == result:
				c[j] += 1
		stand = sum(c) / len(c)
		for j in range(10):
			w[j] = c[j] / (i+1)
		for j in range(10):
			if c[j] > stand:
				w[j] *= 1.2
			else:
				w[j] *= 0.8
	print("weight: ",w)
	return pred

def create_features(size, neg_ratio):
	neg_num = int(size * neg_ratio)
	pos_num = int(size - neg_num)
	#make two features for 1 
	plt.figure(1)
	ax = plt.axes(projection='3d')
	#negative
	# ten features now
	x2, y = make_blobs(n_samples=neg_num, centers=1, n_features=10,
	                   random_state=0, cluster_std=2.1, center_box=(1, 1), shuffle=True)
	ax.scatter3D(x2[:, 0], x2[:, 1], x2[:, 2], 'r.', color='blue', alpha=0.5)
	#positive
	x1, y = make_blobs(n_samples=pos_num, centers=1, n_features=10,
	                   random_state=0, cluster_std=1.6, center_box=(5.5, 5.5), shuffle=True)
	ax.scatter3D(x1[:, 0], x1[:, 1], x1[:, 2], 'r+', color='red', alpha=0.5)
	plt.show()

	correct = [1] * pos_num
	correct = np.concatenate((correct, [-1] * neg_num), axis=0)
	x1.round(decimals=3)
	x2.round(decimals=3)
	x = np.concatenate((x1, x2), axis=0)
	return x, correct


	########################################
	# Adaboost: reference: https://github.com/jaimeps/adaboost-implementation/blob/master/adaboost.py
	#error rate
def get_error_rate(pred, Y):
	return (sum(pred != Y) / float(len(Y)))


def generic_clf(Y_train, X_train, Y_test, X_test, clf, y_true_train, y_true_test):
	clf.fit(X_train, Y_train)
	pred_train = clf.predict(X_train)
	pred_test = clf.predict(X_test)
	return get_error_rate(pred_train, Y_train), \
		get_error_rate(pred_test, Y_test), \
		get_error_rate(pred_train, y_true_train), \
		get_error_rate(pred_test, y_true_test)
#adaboost


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf, Y_true_train, Y_true_test, weight):
	n_train, n_test = len(X_train), len(X_test)
	# Initialize weights
	# w = array([1/n, 1/n, ... 1/n])
	#w = np.ones(n_train) / n_train
	pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
	
	for i in range(M):
		# Fit a classifier with the specific weights
		clf.fit(X_train, Y_train, sample_weight=weight)
		pred_train_i = clf.predict(X_train)
		pred_test_i = clf.predict(X_test)
		# Indicator function
		#miss has 1, 0
		miss = [int(x) for x in (pred_train_i != Y_train)]
		# Equivalent miss with 1,-1 to update weights
		miss2 = [x if x==1 else -1 for x in miss]
		# Error
		err_m = np.dot(weight, miss) / sum(weight)
		# Alpha
		alpha_m = 0.5 * np.log( (1 - err_m) / max(1e-16, float(err_m)))
		# New weights
		weight = np.multiply(weight, np.exp([float(x) * alpha_m for x in miss2]))
		# Add to prediction
		pred_train = [sum(x) for x in zip(pred_train, 
										[x * alpha_m for x in pred_train_i])]
		pred_test = [sum(x) for x in zip(pred_test, 
										[x * alpha_m for x in pred_test_i])]
	
	pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
	
	# Return error rate in train and test set
	return get_error_rate(pred_train, Y_train), \
		get_error_rate(pred_test, Y_test), \
			get_error_rate(pred_train, Y_true_train), \
			get_error_rate(pred_test, Y_true_test), pred_train, pred_test


def plot_error_rate(er_train, er_val, er_test, method):
	df_error = pd.DataFrame([er_train, er_val]).T
	df_error.columns = ['Training_'+method, 'Validation_'+method]
	plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
						color=['lightblue', 'darkblue'], grid=True)
	df = pd.DataFrame({
            'best_it': [(er_test[0]-10)/20 + 1],
            'error_rate': [er_test[1]]
        })
	df.plot(x='best_it', y='error_rate', kind='scatter', color='red',
	           label='the error rate for test data', ax = plot1)
	plt.text(0, 0, "error rate in test data is:" + str(er_test[1]))
	plot1.set_xlabel('Number of iterations', fontsize=12)
	plot1.set_xticklabels(range(0, 450, 50))
	plot1.set_ylabel('Error rate', fontsize=12)
	plot1.set_title('Error rate vs number of iterations\n'+method, fontsize=16)
	plt.axhline(y=er_val[0], linewidth=1, color='red', ls='dashed')
	plt.show()


	########################################
def reorder(arr_train, arr_test, ind_train, ind_test):
	n = len(arr_train) + len(arr_test)
	temp = [0] * n
	# arr[i] should be
	# present at index[i] index
	for i in range(0, len(arr_train)):
		temp[ind_train[i]] = arr_train[i]
	for i in range(0, len(arr_test)):
		temp[ind_test[i]] = arr_test[i]
	return temp

##################################################


if __name__ == '__main__':
	random.seed(9001)
	# x is (x1, x2), tag is 1/-1
	num_samples = 5000
	#####this line is to create random data (used before ver6)
	x, tag = create_features(num_samples, 0.9)
	#indice to keep track of which row of dataset is selected
	indices = range(num_samples)
	# indices_train,indices_test: ([1, 2, 6 ...], [0, 3, 4, 5, ...])
	# since i used random state, it will keep the same

	#train:validation:test = 0.6 : 0.2 : 0.2
	X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
		x, tag, indices, test_size=0.2, random_state=15)
	X_train, X_val, y_train, y_val, ind_train, ind_val = train_test_split(
		X_train, y_train, ind_train, test_size=0.25, random_state=15)
	workbook = xlsxwriter.Workbook('result.xlsx')
	worksheet = workbook.add_worksheet() 
	
	list_experts = [[],[],[],[],[],[],[],[],[],[]]
	list_dif_experts = [0.004, 0.09, 0.1, 0.13, 0.024, 0.07, 0.2, 0.01, 0.18, 0.08]
	for point in x:
		dist_pos = distance.euclidean([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], point)
		dist_neg = distance.euclidean([5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5], point)
		if dist_pos - dist_neg < -4: 
			for i in range(10):
				list_experts[i].append(np.random.choice(
					[-1, 1], p=[0.995-list_dif_experts[i]/5, 0.005+list_dif_experts[i]/5]))
		elif dist_pos - dist_neg < -1:
			for i in range(10):
				list_experts[i].append(np.random.choice([-1, 1], p=[0.97-list_dif_experts[i], 0.03+list_dif_experts[i]]))
		elif dist_pos - dist_neg < 4.5:
			for i in range(10):
				list_experts[i].append(np.random.choice(
					[-1, 1], p=[0.81-list_dif_experts[i], 0.19+list_dif_experts[i]]))
		elif dist_pos - dist_neg < 6:
			for i in range(10):
				list_experts[i].append(np.random.choice(
					[-1, 1], p=[0.65-list_dif_experts[i], 0.35+list_dif_experts[i]]))
		else: 
			for i in range(10):
				list_experts[i].append(np.random.choice(
					[-1, 1], p=[0.01-list_dif_experts[i]/20, 0.99+list_dif_experts[i]/20]))
	#list_experts: [[all prediction by expert1],[ .. expert 2], ...  ]
	#list_sum: [[all experts prediction for the first object][...for 2nd object]]
	list_sum = [list(a) for a in zip(*list_experts)]

	i = 0
	plt.figure(3)
	ax = plt.axes(projection='3d')
	result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	index = 0
	for each in x:
		for i in range(10):
			if list_experts[i][index] == tag[index]:
			 	result[i] += 1
			# if list_experts[i][index] == 1:
			# 	ax.scatter3D(each[0], each[1], each[2], 'r+', color='red', alpha=0.5)
			# if list_experts[i][index] == -1:
			# 	ax.scatter3D(each[0], each[1], each[2], 'r.', color='blue', alpha=0.5)
		index += 1
	print("accuracy: ", [accuracy / len(tag) for accuracy in result])
	# plt.show()
	#### Use truthFinder
	"""
	claims = dict()
	claims['source_id'] = []
	for i in range(5):
		for j in range(5000):
			claims['source_id'].append(i)
	claims['object_id'] = []
	for i in range(5):
		for j in range(5000):
			claims['object_id'].append(j)
	claims['value'] = []
	for i in range(5):
		for j in range(5000):
			claims['value'].append(list_sum[j][i])
	claims = pd.DataFrame(data=claims)
	trust_df, truth_df = truthFinder.truthfinder(claims,
						  imp_func=truthFinder.imp,
						  initial_trust=0.9,
						  similarity_threshold=(1 - 1e-05),
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
	"""
	major_pred = majoriy(list_experts)
	weight_pred = weight(list_experts)
	print("accuracy of majority vote pred: ", accuracy(major_pred, tag),
	      "\naccuracy of weighted: ", accuracy(weight_pred, tag))

	##adjust the initial weight for adaboost
	w = []
	train_len = len(X_train)
	index_w = 0
	for diag in list_sum:
		if index_w in ind_train:
			w.append(10**abs(sum(diag)))
			#print(2**abs(sum(diag)))
		index_w += 1
	norm = [float(i)/sum(w) for i in w]
	#print("the initial weight is: ", norm, "length ", len(norm))
	#set up our inaccurate true labe
	## X_train_maj, X_test_maj, y_train_maj, y_test_maj = train_test_split(
	##     x, major_pred, test_size=0.33, random_state=15)
	X_train_wei, X_test_wei, y_train_wei, y_test_wei= train_test_split(
		x, weight_pred, test_size=0.2, random_state=15)
	X_train_wei, X_val_wei, y_train_wei, y_val_wei = train_test_split(
		X_train_wei, y_train_wei, test_size=0.25, random_state=15)
	## X_train_find, X_test_find, y_train_find, y_test_find = train_test_split(
	##         x, find_pred, test_size=0.33, random_state=15)
	#Use adaboost for major and weghted
	clf_tree = DecisionTreeClassifier(max_depth=3, random_state=1)
	## er_tree_maj = generic_clf(y_train_maj, X_train_maj,
	##                           y_test_maj, X_test_maj, clf_tree, y_train, y_test)
	er_tree_wei = generic_clf(y_train_wei, X_train_wei,
							  y_val_wei, X_val_wei, clf_tree, y_train, y_val)
	baseline = generic_clf(y_train_wei, X_train_wei,
                        y_test_wei, X_test_wei, clf_tree, y_train, y_test)
	print("----baseline using decision tree----\nthe accurary is:", str(1-baseline[3]), "\n----------------")
	## er_tree_find = generic_clf(y_train_find, X_train_find, y_test_find, X_test_find, clf_tree, y_train, y_test)
	# Fit Adaboost classifier using a decision tree as base estimator
	# Test with different number of iterations
	## er_train_maj, er_test_maj = [er_tree_maj[0]], [er_tree_maj[1]]
	er_train_wei, er_val_wei, er_true_train_wei, er_true_val_wei = [er_tree_wei[0]], [er_tree_wei[1]], [er_tree_wei[2]], [er_tree_wei[3]]
	## er_train_find, er_test_find = [er_tree_find[0]], [er_tree_find[1]]
	## er_true_train_maj, er_true_test_maj = [er_tree_maj[2]], [er_tree_maj[3]]
	##er_true_train_find, er_true_test_find = [er_tree_find[2]], [er_tree_find[3]]
	#x_range = 10, 35, 60, 85, ... 410
	x_range = range(10, 450, 20)
	low_error = 1
	best_iter = 0
	for i in x_range:
		clf = GradientBoostingClassifier(n_estimators = i, random_state = 15)
		clf.fit(X_train_wei, y_train_wei, sample_weight= norm)
		## er_i_maj = adaboost_clf(y_train_maj, X_train_maj,y_test_maj, X_test_maj, i, clf_tree, y_train, y_test, weight = w)
		## er_train_maj.append(er_i_maj[0])
		## er_test_maj.append(er_i_maj[1])
		## er_true_train_maj.append(er_i_maj[2])
		## er_true_test_maj.append(er_i_maj[3])
		## er_i_wei = adaboost_clf(y_train_wei, X_train_wei,y_val_wei, X_val_wei, i, clf_tree, y_train, y_val, weight = w)
		pred_i_wei_val = clf.predict(X_val_wei)
		pred_i_wei_train = clf.predict(X_train_wei)
		er_train_wei.append(get_error_rate(pred_i_wei_train, y_train_wei))
		er_val_wei.append(get_error_rate(pred_i_wei_val, y_val_wei))
		er_true_train_wei.append(get_error_rate(pred_i_wei_train, y_train))
		er_true_val_wei.append(get_error_rate(pred_i_wei_val, y_val))
		## er_i_find = adaboost_clf(y_train_find, X_train_find,
		##                         y_test_find, X_test_find, i, clf_tree, y_train, y_test, weight=w)
		## er_train_find.append(er_i_find[0])
		## er_test_find.append(er_i_find[1])
		## er_true_train_find.append(er_i_find[2])
		## er_true_test_find.append(er_i_find[3])
		if(low_error > er_val_wei[-1]):  # this is compared to our y
			low_error = er_val_wei[-1]
			best_iter = i
	clf = GradientBoostingClassifier(n_estimators=best_iter, random_state=15)
	clf.fit(X_train_wei, y_train_wei, sample_weight=norm)
	pred_train = clf.predict(X_train_wei)
	pred_test = clf.predict(X_test_wei)
	er_true_test_wei = get_error_rate(pred_test, y_test)
	er_test_wei = get_error_rate(pred_test, y_test_wei)
	## pred_train = er_i_find[4]
	## pred_test = er_i_find[5]
	# Compare error rate vs number of iterations
	#plot_error_rate(er_train_maj, er_test_maj,"Majority vote label used in adaboost and calculating error")
	#plot_error_rate(er_train_wei, er_test_wei,"Weighting vote label used in adaboost and calculating error ")
	## plot_error_rate(er_true_train_maj, er_true_test_maj,
	## 				"Majority vote label for adaboost; real true label as error detector")
	#plot_error_rate(er_train_wei, er_val_wei, [best_iter, er_test_wei], "Weighted vote label for gradient (compare to predicted label)")
	plot_error_rate(er_true_train_wei, er_true_val_wei, [best_iter, er_true_test_wei],
                 "Weighted vote label for gradient (compare to true label)")

##############################################
	er_train_wei_l, er_val_wei_l = [er_tree_wei[0]], [er_tree_wei[1]]
	er_true_train_wei_l, er_true_val_wei_l = [er_tree_wei[2]], [er_tree_wei[3]]
	learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
	low_error = 1
	best_lr = 0
	best_lr_index = 0
	index = -1
	for eta in learning_rates:
		index += 1
		clf = GradientBoostingClassifier(n_estimators=best_iter, random_state=15, learning_rate=eta)
		clf.fit(X_train_wei, y_train_wei, sample_weight=norm)
		pred_i_wei_val = clf.predict(X_val_wei)
		pred_i_wei_train = clf.predict(X_train_wei)

		er_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train_wei))
		er_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val_wei))
		er_true_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train))
		er_true_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val))
		if(low_error > er_val_wei_l[-1]):  # this is compared to our y
			low_error = er_val_wei_l[-1]
			best_lr = eta
			best_lr_index=index
	
	clf = GradientBoostingClassifier(n_estimators=best_iter, random_state=15, learning_rate = best_lr)
	clf.fit(X_train_wei, y_train_wei, sample_weight=norm)
	pred_train = clf.predict(X_train_wei)
	pred_test = clf.predict(X_test_wei)
	er_true_test_wei = get_error_rate(pred_test, y_test)
	er_test_wei = get_error_rate(pred_test, y_test_wei)
	print("Using the best (iteration, learning rate)= ("+ str(best_iter) + ", " + str(best_lr) + "),\nthe error rate is: ", er_true_test_wei)
	
	"""
	df_error = pd.DataFrame([er_true_train_wei_l, er_true_val_wei_l]).T
	df_error.columns = ['Training_', 'Validation_']
	plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                       color=['lightblue', 'darkblue'], grid=True)
	df = pd.DataFrame({
               'best_lr': [best_lr_index],
               'error_rate': [er_true_test_wei]
               })
	df.plot(x='best_lr', y='error_rate', kind='scatter', color='red',
         label='the error rate for test data', ax=plot1)
	plot1.set_xlabel('learning rates', fontsize=12)
	plot1.set_xticklabels(learning_rates)
	plot1.set_ylabel('Error rate', fontsize=12)
	plot1.set_title('Error rate vs learning rates\n', fontsize=16)
	plt.show()
	"""
	## plot_error_rate(er_true_train_find, er_true_test_find,
	## 				"TruthFind label for adaboost; real true label as error detector")

	########################################
	
	##############
	#building models
	#clf = GaussianNB()
	#clf = KNeighborsClassifier()
	#clf = tree.DecisionTreeClassifier()
	

##############################

	# Create adaboost classifer object
	#svc=SVC(probability=True, kernel='linear')
	clf_ttt = GradientBoostingClassifier(
		n_estimators=best_iter, random_state=15, learning_rate=best_lr)

	abc = AdaBoostClassifier(n_estimators=300, base_estimator=DecisionTreeClassifier(
		max_depth=3), learning_rate=1)
	offi_wei = abc.fit(X_train_wei, y_train_wei, sample_weight= None)
	weight_wei = abc.fit(X_train_wei, y_train_wei, sample_weight = norm)
	y_wei_train_pred_ada = offi_wei.predict(X_train_wei)
	y_wei_val_pred_ada = offi_wei.predict(X_val_wei)
	er_train_wei = get_error_rate(y_wei_train_pred_ada, y_train_wei)
	er_val_wei = get_error_rate(y_wei_val_pred_ada, y_test_wei)
	print("train eror rate", er_train_wei, "test error eate", er_val_wei)
	y_wei_train_pred_ada = weight_wei.predict(X_train_wei)
	y_wei_val_pred_ada = weight_wei.predict(X_val_wei)
	er_train_wei = get_error_rate(y_wei_train_pred_ada, y_train_wei)
	er_val_wei = get_error_rate(y_wei_val_pred_ada, y_test_wei)
	print("train eror rate with changed weight", er_train_wei, "test error eate", er_val_wei)
	"""
	plot_error_rate(er_train_wei, er_test_wei,
				 "offi: Weighting vote label for adaboost & calculate error ")
	plot_error_rate(er_true_train_maj, er_true_test_maj,
				 "offi: Majority vote label for adaboost; real true label as error detector")
	plot_error_rate(er_true_train_wei, er_true_test_wei,
				 "offi: Weighting vote label for adaboost; real true label as error detector")
	"""

	fpr, tpr, thresholds = metrics.roc_curve(y_train_wei, train_pred)
	roc_auc = auc(fpr, tpr)
	train_results.append(roc_auc)

##############################
	"""
	prec = []
	recall = []
	recall_auc = []
	fpr = []
	tpr = []
	roc_auc = []
	y_preds = [y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y_major_pred, y_wei_pred, y_major_pred_ada]
	for i in range(8):
		p, re, thresholds = metrics.precision_recall_curve(y_test, y_preds[i], pos_label=1)
		prec.append(p)
		recall.append(re)
		auc = metrics.auc(recall[i], prec[i])
		recall_auc.append(auc)
		a, b, thresholds = metrics.roc_curve(y_test, y_preds[i], pos_label=1)
		fpr.append(a)
		tpr.append(b)
		aucs = metrics.auc(fpr[i], tpr[i]) 
		roc_auc.append(aucs)
	

	lw = 2
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', "hotpink", "green", "red", "darkred", "pink"])
	for i, color in zip(range(8), colors):
		plt.plot(recall[i], prec[i], color=color, lw=lw,
				 label='ROC curve of class {0} (area = {1:0.2f})'
				 ''.format(i, recall_auc[i]))
	
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.title('ROC curve for five experts')
	plt.legend(loc="lower right")
	plt.show()

######################################################
	"""
	row = 0
	column = 0
	worksheet.write_row('A1', ["x1", "x2", "correct tag", "e1", "e2", "e3", "e4", "e5", "ada_pred"])
	row += 1
	# iterating through content list 
	ada_pred = reorder(pred_train, pred_test, ind_train, ind_test)
	for i in range(num_samples): 
		column = 0
		# write operation perform 
		worksheet.write_row(row, column, [x[i][0], x[i][1], tag[i], e1[i], e2[i], e3[i], e4[i], e5[i]])
		row += 1
	#worksheet.write(row, 0, "major_accuracy: "+ str(y_major_ac))
	#worksheet.write(row+1, 0, "weight_accuracy: "+ str(y_wei_ac))

	
	workbook.close() 


#reference
#https://web.stanford.edu/~hastie/Papers/samme.pdf
#https://github.com/totucuong/spectrum/blob/master/spectrum/judge/truthfinder.py
