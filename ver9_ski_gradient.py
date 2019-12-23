import numpy as np
from numpy.linalg import cholesky
import xlsxwriter
import matplotlib.pyplot as plt
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
import random
import truthFinder 
from itertools import cycle
from sklearn import tree
import pandas as pd

#v6 new changes: consider the relationship between our labels and the features
#v7 new changes: initial weight for adaboost changed
#v8 new changes: used the truthdiscovery (truth finder)
#v9 new changes: only left "weighted" option. 
# 				Saved the result for different initial weights
#				Added a validation dataset to determine number of iteration
# this version: used GradientBoostingClassifier from sklearn
#from truthdiscovery import TruthFinder


# #for future improvement
# 1. change the weight and plot 
# 2. read paper for gradient tree boosting
# 3. not only 5 experts;  the possibility 
# 4. snorkle to find labeling. Crowd sourcing 
#  covarience the same 
#  add more features 
#  add noisy labels
# 5. linnear discripmi analysis, descrminate analysis 
# 6. Try Logistic, De Tree, Random Forest, SVM
# 7. Add graph: Error , Presision, Recall, AUPRC
# 8. Compare pla
# 9. the best iteration rate is the earlist lowest rate. 
# 10. Stochastic 

#majority input: decisions from all experts
#         output: prediction. 
def majoriy(e1, e2, e3, e4, e5):
	pred = []
	for i in range(len(e1)):
		if (e1[i] + e2[i] + e3[i] + e4[i] + e5[i] < 0):
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

def weight(e1, e2, e3, e4, e5):
	# weight
	w1 = 1
	w2 = 1
	w3 = 1
	w4 = 1
	w5 = 1
	#count 
	c1 = 0
	c2 = 0
	c3 = 0
	c4 = 0
	c5 = 0
	pred = []
	for i in range(len(e1)):
		#print(w1, w2, w3)
		#print(c1, c2, c3)
		if(w1 * e1[i] + w2 * e2[i] + w3 * e3[i] + w4 * e4[i] + w5 * e5[i] > 0):
			pred.append(1)
			if e1[i] == 1: 
				c1 += 1
			if e2[i] == 1: 
				c2 += 1
			if e3[i] == 1: 
				c3 += 1
			if e4[i] == 1: 
				c4 += 1
			if e5[i] == 1: 
				c5 += 1
		else:
			pred.append(-1)
			if e1[i] == -1: 
				c1 += 1
			if e2[i] == -1: 
				c2 += 1
			if e3[i] == -1: 
				c3 += 1
			if e4[i] == -1: 
				c4 += 1
			if e5[i] == -1: 
				c5 += 1
		w1 = c1 / (i+1) 
		w2 = c2 / (i+1) 
		w3 = c3 / (i+1) 
		w4 = c4 / (i+1) 
		w5 = c5 / (i+1) 
	print("weight: ",w1, w2, w3, w4, w5)
	return pred

def create_features(size, neg_ratio):
	neg_num = int(size * neg_ratio)
	pos_num = int(size - neg_num)
	#make two features for 1 
	plt.figure(1)

	#negative
	mu2 = np.array([[1.3, 1.3]])
	Sigma2 = np.array([[1.2, 1], [0.9, 1.8]])
	R2 = cholesky(Sigma2)
	np.random.seed(415)
	x2 = np.dot(np.random.randn(neg_num, 2), R2) + mu2
	plt.plot(x2[:,0],x2[:,1],'r.', color = 'blue', alpha=0.5)
	#positive
	mu1 = np.array([[5, 4]])
	Sigma1 = np.array([[1.2, 1], [0.75, 1.5]])
	R1 = cholesky(Sigma1)
	x1 = np.dot(np.random.randn(pos_num, 2), R1) + mu1
	plt.plot(x1[:, 0], x1[:, 1], 'r+', color='red', alpha=0.5)
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
	# indices_train,indices_test: ([1, 2, ...], [0, 3, ...])
	# since i used random state, it will keep the same
	X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
		x, tag, indices, test_size=0.2, random_state=15)
	X_train, X_val, y_train, y_val, ind_train, ind_val = train_test_split(
		X_train, y_train, ind_train, test_size=0.25, random_state=15)
	workbook = xlsxwriter.Workbook('result.xlsx')
	worksheet = workbook.add_worksheet() 
	
	# y = -1.46x +3.75: y smaller than this is guarenteed negative
	# y = −0.2506986027944111x + 2.983626347305389: y smaller than this is 98% neg
	# y = −1.4112450758027875x + 12.026225379013937: y smaller than this unknown
	# y larger is guarenteed pos
	e1 = []
	e2 = []
	e3 = []
	e4 = []
	e5 = []
	list_experts = [e1, e2, e3, e4, e5]
	for x0, x1 in x:
		if -1.46*x0 + 3.75 > x1:
			for i in range(5):
				list_experts[i].append(np.random.choice([-1, 1], p=[0.995, 0.005]))
		elif -0.2506986027944111*x0 + 2.9836263473054 > x1:
			list_experts[0].append(np.random.choice([-1, 1], p=[0.99, 0.01]))
			list_experts[1].append(np.random.choice([-1, 1], p=[0.98, 0.02]))
			list_experts[2].append(np.random.choice([-1, 1], p=[0.98, 0.02]))
			list_experts[3].append(np.random.choice([-1, 1], p=[0.93, 0.07]))
			list_experts[4].append(np.random.choice([-1, 1], p=[0.92, 0.08]))
		elif -0.9312908502013819 * x0 + 7.532693738791448 > x1:
			list_experts[0].append(np.random.choice([-1, 1], p=[0.96, 0.04]))
			list_experts[1].append(np.random.choice([-1, 1], p=[0.94, 0.06]))
			list_experts[2].append(np.random.choice([-1, 1], p=[0.92, 0.08]))
			list_experts[3].append(np.random.choice([-1, 1], p=[0.89, 0.11]))
			list_experts[4].append(np.random.choice([-1, 1], p=[0.87, 0.13]))
		elif - 1.4112450758027875 * x0 + 12.026225379013937 > x1:
			list_experts[0].append(np.random.choice([-1, 1], p=[0.81, 0.19]))
			list_experts[1].append(np.random.choice([-1, 1], p=[0.79, 0.21]))
			list_experts[2].append(np.random.choice([-1, 1], p=[0.78, 0.22]))
			list_experts[3].append(np.random.choice([-1, 1], p=[0.75, 0.25]))
			list_experts[4].append(np.random.choice([-1, 1], p=[0.73, 0.27]))
		else:
			for i in range(5):
				list_experts[i].append(np.random.choice([-1, 1], p=[0.01, 0.99]))
	
	list_sum = [list(a) for a in zip(e1, e2, e3, e4, e5)]
	# i = 2
	# plt.figure(i+2)
	# index = 0
	# for x1, x2 in x:
	# 	if list_experts[i][index] == 1:
	# 		plt.plot(x1, x2, 'r+', color='red', alpha=0.5)
	# 	if list_experts[i][index] == -1:
	# 		plt.plot(x1, x2, 'r.', color='blue', alpha=0.5)
	# 	index += 1
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
	## major_pred = majoriy(e1, e2, e3, e4, e5)
	weight_pred = weight(e1, e2, e3, e4, e5)
	##adjust the weight 
	w = []
	train_len = len(X_train)
	index_w = 0
	for diag in list_sum:
		if index_w in ind_train:
			if(sum(diag) == 5 or sum(diag) == -5):
				w.append(100000)
			if (sum(diag) == 3 or sum(diag) == -3):
				w.append(100)
			if (sum(diag) == 1 or sum(diag) == -1):
				w.append(1)
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
	## er_tree_find = generic_clf(y_train_find, X_train_find, y_test_find, X_test_find, clf_tree, y_train, y_test)
	# Fit Adaboost classifier using a decision tree as base estimator
	# Test with different number of iterations
	## er_train_maj, er_test_maj = [er_tree_maj[0]], [er_tree_maj[1]]
	er_train_wei, er_val_wei = [er_tree_wei[0]], [er_tree_wei[1]]
	## er_train_find, er_test_find = [er_tree_find[0]], [er_tree_find[1]]
	## er_true_train_maj, er_true_test_maj = [er_tree_maj[2]], [er_tree_maj[3]]
	er_true_train_wei, er_true_val_wei = [er_tree_wei[2]], [er_tree_wei[3]]
	##er_true_train_find, er_true_test_find = [er_tree_find[2]], [er_tree_find[3]]
	#x_range = 10, 35, 60, 85, ... 410
	x_range = range(10, 450, 20)
	low_error = 1
	best_iter = 0
	flag = False # to see if there is a vibration in error
	for i in x_range:
		clf = GradientBoostingClassifier(n_estimators = i, random_state = 15)
		clf.fit(X_train_wei, y_train_wei, sample_weight= norm)
		## er_i_maj = adaboost_clf(y_train_maj, X_train_maj,
		##                         y_test_maj, X_test_maj, i, clf_tree, y_train, y_test, weight = w)
		## er_train_maj.append(er_i_maj[0])
		## er_test_maj.append(er_i_maj[1])
		## er_true_train_maj.append(er_i_maj[2])
		## er_true_test_maj.append(er_i_maj[3])
		## er_i_wei = adaboost_clf(y_train_wei, X_train_wei,
		## 						y_val_wei, X_val_wei, i, clf_tree, y_train, y_val, weight = w)
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
			flag = False
		elif (low_error == er_val_wei[-1]):
			if flag == True:
				best_iter = i
		else:
			flag = True
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
	

	plot_error_rate(er_train_wei, er_val_wei, [best_iter, er_test_wei],
                 "Weighting vote label for gradient (compare to predicted label)")
	plot_error_rate(er_true_train_wei, er_true_val_wei, [best_iter, er_true_test_wei],
                 "Weighting vote label for gradient (compare to true label)")

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

		fpr, tpr, thresholds = roc_curve(y_train_wei, train_pred)
		roc_auc = auc(fpr, tpr)
		train_results.append(roc_auc)

		er_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train_wei))
		er_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val_wei))
		er_true_train_wei_l.append(get_error_rate(pred_i_wei_train, y_train))
		er_true_val_wei_l.append(get_error_rate(pred_i_wei_val, y_val))
		if(low_error > er_val_wei_l[-1]):  # this is compared to our y
			low_error = er_val_wei_l[-1]
			best_lr = eta
			print("best", best_lr)
			best_lr_index = index
	
	clf = GradientBoostingClassifier(n_estimators=best_iter, random_state=15, learning_rate = best_lr)
	clf.fit(X_train_wei, y_train_wei, sample_weight=norm)
	pred_train = clf.predict(X_train_wei)
	pred_test = clf.predict(X_test_wei)
	er_true_test_wei = get_error_rate(pred_test, y_test)
	er_test_wei = get_error_rate(pred_test, y_test_wei)
	
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
	abc = AdaBoostClassifier(n_estimators=300, base_estimator=DecisionTreeClassifier(
		max_depth=3), learning_rate=1)
	offi_wei = abc.fit(X_train_wei, y_train_wei)
	y_wei_train_pred_ada = offi_wei.predict(X_train_wei)
	y_wei_val_pred_ada = offi_wei.predict(X_val_wei)
	er_train_wei = get_error_rate(y_wei_train_pred_ada, y_train_wei)
	er_val_wei = get_error_rate(y_wei_val_pred_ada, y_test_wei)
	print("train eror rate", er_train_wei, "test error eate", er_val_wei)
	"""
	plot_error_rate(er_train_wei, er_test_wei,
				 "offi: Weighting vote label for adaboost & calculate error ")
	plot_error_rate(er_true_train_maj, er_true_test_maj,
				 "offi: Majority vote label for adaboost; real true label as error detector")
	plot_error_rate(er_true_train_wei, er_true_test_wei,
				 "offi: Weighting vote label for adaboost; real true label as error detector")
	"""

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
