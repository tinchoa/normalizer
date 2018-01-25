from sklearn import svm
import csv
import numpy as np
from newmain import NewMain 
from sklearn.metrics import accuracy_score

def correlationCalculate(vectors):

	cof=np.corrcoef(np.array(vectors),rowvar=False)
	w={}
	aij={}
	variancia=np.var(np.nan_to_num(cof),0)
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
	
	return reduced,index

def MatrixReducer(vectors, index):
	reducedMatrix =[]
	vectors = np.matrix(vectors)

	for k in index:
		reducedMatrix.append(vectors[:,k]) #reduced matrix 

	vectors2 = np.column_stack(reducedMatrix)
	vectors2 = np.array(vectors2)
	
	return vectors2

def traininML(linhas):
	'''
	normalize,feature selection and SMV model training
	'''

	label=[]
	dados =[]
	for linha in linhas:
		a=linha.split(',')
		label.append(float(a[len(a)-1].split('\n')[0]))
		dados.append(np.asfarray(a[5:len(a)-2]))#removing IPsrc,IPdst,portsrc,portdsc,proto


	'''
	normalize the data (with our implementation)
	'''
	print 'normalizing...'
	normalize=NewMain().run(dados,0)
	print 'done...'
	'''
	reduced with feature selection
	'''

	reduced,index=correlationCalculate(normalize)

	'''
	train SVM model
	'''	
	clf = svm.SVC()

	#dadosNorm=


	model=clf.fit(reduced, label)
	

	return model,index


train = open("classes-17-reduced.out", "r")

linhas=train.readlines()

modelo,index=traininML(linhas)

print 'modelo criado'

features={} #to see the evolution of the features

label=[]
dados =[]
window=0
tamanhoJanela=500

data = open("classes-17-end.out", "r")
linha = data.readline()
while linha !="":
	janela = []
	label=[]
	while linha !="" and len(janela) < tamanhoJanela:
		tmp1 = linha.strip("\n").split(",")[5:-2]#removing IPsrc,IPdst,portsrc,portdsc,proto,class
		tmp2 = []
		for i in tmp1:
			tmp2.append(float(i))
		janela.append(tmp2)
		label.append(float(linha.strip("\n").split(",")[-1]))
		linha = data.readline()

	normalize=NewMain().run(janela,0)
	#reduced,index=correlationCalculate(normalize)
	#features[window]=index
	reduced=MatrixReducer(normalize,index)
	classification=modelo.predict(reduced)
	acc=accuracy_score(label,classification,normalize==True)
	print acc
	window+=1
	