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
from sklearn import metrics
from sklearn.metrics import auc   
from sklearn.svm import SVC 

from itertools import cycle

from sklearn import tree
import pandas as pd

from truthdiscovery import TruthFinder


#majority input: decisions from all experts
#         output: prediction. 
def majoriy(e1, e2, e3, e4, e5):
	pred = []
	for i in range(500):
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
	for i in range(500):
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
	mu = np.array([[5, 4]])
	Sigma = np.array([[1.2, 1], [0.75, 1.5]])
	R = cholesky(Sigma)
	x1 = np.dot(np.random.randn(pos_num, 2), R) + mu
	plt.subplot(221)
	plt.plot(x1[:,0],x1[:,1],'r+')
	interactive(True)
	plt.show()
	#make two features for 0
	#plt.figure(2)
	mu = np.array([[1.3, 1.3]])
	Sigma = np.array([[1.2, 1], [0.9, 1.8]])
	R = cholesky(Sigma)
	x2 = np.dot(np.random.randn(neg_num, 2), R) + mu
	plt.subplot(222)
	plt.plot(x2[:,0],x2[:,1],'r.')
	interactive(False)
	plt.show()

	correct = np.ones((pos_num, 1), dtype=int)
	correct = np.concatenate((correct, np.full((neg_num, 1), int(-1))), axis=0)
	x = np.concatenate((x1, x2), axis=0)
	return x, correct

if __name__ == '__main__':

	# x is (x1, x2), tag is 1/-1
	x, tag = create_features(500, 0.9)
	X_train, X_test, y_train, y_test = train_test_split(x, tag, test_size=0.33, random_state=15)

	workbook = xlsxwriter.Workbook('result.xlsx') 
	worksheet = workbook.add_worksheet() 

	e1 = np.random.choice([-1, 1], size=500, p=[.9, .1])
	e2 = np.random.choice([-1, 1], size=500, p=[.75, .25])
	e3 = np.random.choice([-1, 1], size=500, p=[.8, .2])
	e4 = np.random.choice([-1, 1], size=500, p=[.85, .15])
	e5 = np.random.choice([-1, 1], size=500, p=[.8, .2])
	major_pred = majoriy(e1, e2, e3, e4, e5)
	weight_pred = weight(e1, e2, e3, e4, e5)



	df1 = pd.DataFrame({'experts':["e1"]*500,'candidate': [str(i) for i in range(500)], 'predicted_tag' : [str(e1[i]) for i in range(500)] })
	df2 = pd.DataFrame({'experts':["e2"]*500,'candidate':[str(i) for i in range(500)], 'predicted_tag' : [str(e2[i]) for i in range(500)] })
	df3 = pd.DataFrame({'experts':["e3"]*500,'candidate':[str(i) for i in range(500)], 'predicted_tag' : [str(e3[i]) for i in range(500)] })
	df4 = pd.DataFrame({'experts':["e4"]*500,'candidate': [str(i) for i in range(500)], 'predicted_tag' : [str(e4[i]) for i in range(500)] })
	df5 = pd.DataFrame({'experts':["e5"]*500,'candidate':[str(i) for i in range(500)], 'predicted_tag' : [str(e5[i]) for i in range(500)] })
	df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
	
	print(df)
	########################################

	vectorizer = TfidfVectorizer(min_df=1)
	vectorizer.fit(df["candidate"])

	def similarity(w1, w2):
	    V = vectorizer.transform([w1, w2])
	    v1, v2 = np.asarray(V.todense())
	    return np.dot(v1, v2) / (norm(v1) * norm(v2))


	def implication(f1, f2):
	    return similarity(f1, f2)


	finder = TruthFinder(implication, dampening_factor=0.8, influence_related=0.6)

	print("Inital state")
	print(df)
	df = finder.train(df)

	print("Estimation result")
	print(df)



	########################################

	##############
	#building models
	#clf = GaussianNB()
	clf = KNeighborsClassifier()
	#clf = tree.DecisionTreeClassifier()

	y1_pred = clf.fit(X_train, e1[:335]).predict(X_test)
	y2_pred = clf.fit(X_train, e2[:335]).predict(X_test)
	y3_pred = clf.fit(X_train, e3[:335]).predict(X_test)
	y4_pred = clf.fit(X_train, e4[:335]).predict(X_test)
	y5_pred = clf.fit(X_train, e5[:335]).predict(X_test)

	y_major_pred = clf.fit(X_train, major_pred[:335]).predict(X_test)
	y_wei_pred = clf.fit(X_train, weight_pred[:335]).predict(X_test)


##############################

	train_adaboost = X_train
	for i in range(4):
		train_adaboost = np.concatenate((train_adaboost, X_train), axis=0)

	y_sum = np.append(e1[:335], e2[:335])
	y_sum = np.append(np.append(y_sum, e3[:335]), e4[:335])
	y_sum = np.append(y_sum, e5[:335])

	# Create adaboost classifer object

	svc=SVC(probability=True, kernel='linear')
	abc =AdaBoostClassifier(n_estimators=100, base_estimator=svc, learning_rate=1)
	model = abc.fit(train_adaboost, y_sum)
	y_major_pred_ada = model.predict(X_test)


##############################


	y1_ac = accuracy(y1_pred, y_test)
	y2_ac = accuracy(y2_pred, y_test)
	y3_ac = accuracy(y3_pred, y_test)
	y4_ac = accuracy(y4_pred, y_test)
	y5_ac = accuracy(y5_pred, y_test)
	y_major_ac = accuracy(y_major_pred, y_test)
	y_wei_ac = accuracy(y_wei_pred, y_test)
	y_major_pred_ada_ac = accuracy(y_major_pred_ada, y_test)

	print("accuracy", y1_ac, y2_ac, y3_ac, y4_ac, y5_ac, y_major_ac,y_wei_ac,y_major_pred_ada_ac)
	

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
	    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	             label='ROC curve of class {0} (area = {1:0.2f})'
	             ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for five experts')
	plt.legend(loc="lower right")
	plt.show()


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

	row = 0
	column = 0
	worksheet.write_row('A1', ["x1", "x2", "correct tag", "e1", "e2", "e3", "e4", "e5"])
	row += 1
	# iterating through content list 
	for i in range(500): 
		column = 0
		# write operation perform 
		worksheet.write_row(row, column, [x[i][0], x[i][1], tag[i], e1[i], e2[i], e3[i], e4[i], e5[i]])
		row += 1
	#worksheet.write(row, 0, "major_accuracy: "+ str(y_major_ac))
	#worksheet.write(row+1, 0, "weight_accuracy: "+ str(y_wei_ac))


	workbook.close() 
