from sklearn import svm
import csv
import numpy as np
from newmain import NewMain 


def correlationCalculate(vectors):

	cof=np.corrcoef(np.array(vectors),rowvar=False)
	w={}
	aij={}
	variancia=np.var(cof,0)
	for i in range(len(cof[0])):
		w[i]=0
		aij[i]=0
		for j in np.nan_to_num(cof[i]):
			k=abs(j)
			aij[i]=aij[i]+k
		w[i]=variancia[i]/aij[i]
	ja=sorted([(value,key) for (key,value) in w.items()],reverse=True)


	index=[]
	for i in ja:
		index.append(i[1])
	
	index=index[0:6] #tacking the first 6 features

	reduced=MatrixReducer(vectors,index)
	
	return reduced



def MatrixReducer(vectors, index):
	reducedMatrix =[]
	vectors = np.matrix(vectors)

	for k in index:
		reducedMatrix.append(vectors[:,k]) #reduced matrix 

	vectors2 = np.column_stack(reducedMatrix)
	vectors2 = np.array(vectors2)
	
	return vectors2



def traininML():

	data = open("classes-17.out", "r")

	linhas=data.readlines()


	label=[]
	dados =[]
	for linha in linhas:
		a=linha.split(',')
		label.append(float(a[len(a)-1].split('\n')[0]))
		dados.append(np.asfarray(a[5:len(a)-2]))#removing IPsrc,IPdst,portsrc,portdsc,proto

	'''
	normalize the data (with our implementation)
	'''
	normalize=NewMain().run(dados,0)
	'''
	reduced with feature selection
	'''
	reduced=correlationCalculate(normalize)

	'''
	train SVM model
	'''	
	clf = svm.SVC()

	#dadosNorm=


	model=clf.fit(reduced, label)
	

	return model

