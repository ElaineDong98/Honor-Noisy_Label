import numpy as np
import xlsxwriter


def majoriy(e1, e2, e3, e4, e5):
	pred = []
	for i in range(500):
		count0 = 0
		if (e1[i] + e2[i] + e3[i] + e4[i] + e5[i] < 0):
			pred.append(-1)
		else:
			pred.append(1)
	return pred


def accuracy(pred, ans):
	count = 0
	for i in range(500):
		if(pred[i] == ans[i]):
			count += 1
	return count/500.0

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
		w1 = c1 * c1 / (i+1) 
		w2 = c2 *c2 / (i+1) 
		w3 = c3 *c3 / (i+1) 
		w4 = c4 *c4 / (i+1) 
		w5 = c5 *c5 / (i+1) 
	print(w1, w2, w3, w4, w5)
	return pred



if __name__ == '__main__':
	workbook = xlsxwriter.Workbook('result.xlsx') 
	worksheet = workbook.add_worksheet() 

	e1 = np.random.choice([-1, 1], size=500, p=[.05, .95])
	e2 = np.random.choice([-1, 1], size=500, p=[.1, .9])
	e3 = np.random.choice([-1, 1], size=500, p=[.3, .7])
	e4 = np.random.choice([-1, 1], size=500, p=[.4, .6])
	e5 = np.random.choice([-1, 1], size=500, p=[.2, .8])
	truth = np.random.choice([-1, 1], size=500, p=[0, 1])
	major_pred = majoriy(e1, e2, e3, e4, e5)
	major_accurate = accuracy(major_pred, truth)
	weight_pred = weight(e1, e2, e3, e4, e5)
	weight_accurate = accuracy(weight_pred, truth)


	row = 0
	column = 0
	worksheet.write(row, column, "e1")
	worksheet.write(row, column+1, "e2") 
	worksheet.write(row, column+2, "e3") 
	worksheet.write(row, column+3, "e4") 
	worksheet.write(row, column+4, "e5") 
	worksheet.write(row, column+5, "truth") 
	worksheet.write(row, column+6, "major_pred")
	worksheet.write(row, column+7, "weight_pred")

	row += 1
	# iterating through content list 
	for i in range(500): 
		column = 0
		# write operation perform 
		worksheet.write(row, column, e1[i])
		worksheet.write(row, column+1, e2[i]) 
		worksheet.write(row, column+2, e3[i])  
		worksheet.write(row, column+3, e4[i])  
		worksheet.write(row, column+4, e5[i]) 
		worksheet.write(row, column+5, truth[i]) 
		worksheet.write(row, column+6, major_pred[i])
		worksheet.write(row, column+7, weight_pred[i])

		# incrementing the value of row by one 
		# with each iteratons. 
		row += 1
	worksheet.write(row, 0, "major_accuracy: "+ str(major_accurate))
	worksheet.write(row+1, 0, "weight_accuracy: "+ str(weight_accurate))


	workbook.close() 
