import numpy as np
import scipy.stats as st
import math
from sklearn import preprocessing

class Original:
	#def __init__(self):
	#	super(Original, self).__init__()

	def run(self):
		def dataPrepare(item):
			''' get the values, remove the categorical data'''
			a=item.split(',')
			label=a[len(a)-1].split('\n')[0]
			data=a[5:len(a)-1]#removing IPsrc,IPdst,portsrc,portdsc,proto
			return data

		def calculate(dataset):
			original=np.asfarray(dataset)
			original=original.tolist()
			X_normalized = preprocessing.normalize(original, norm='max')
			lower, upper = -3.09, 3.09
			X_normed = [lower + (upper - lower) * x for x in X_normalized]
			return X_normalized
		
		np.set_printoptions(precision=3)
		np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})
		files=open('classes-17.out','r')
		lines=files.readlines()
		self.batch=[]
		batch=self.batch
		for i in lines:
			batch.append(dataPrepare(i))

		return calculate(batch)


	




