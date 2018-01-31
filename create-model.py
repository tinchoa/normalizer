from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
import csv
import numpy as np
from newmain import NewMain 
from maxMin_Normalizer import maxMin_Normalizer

'''
normalizer do sklearner
'''
from sklearn.preprocessing import Normalizer


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE #to balance training dataset

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
	
	index=index[0:8] #tacking the first 6 features

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

def dataPreparing(linhas):
	label=[]
	dados =[]
	for linha in linhas:
		a=linha.split(',')
		tmp=float(a[len(a)-1].split('\n')[0])#label
		if tmp == 0.0:
			tmp=1.0
		else:
			tmp=-1.0
		label.append(tmp)
		dados.append(np.asfarray(a[5:len(a)-2]))#removing IPsrc,IPdst,portsrc,portdsc,proto
	return dados,label

def dataSampling(dados,label):

	sm = SMOTE(ratio='minority') #to sample data
	dadosSample,labelSample=sm.fit_sample(dados,label)

	return dadosSample,labelSample

def traininML(dados,label):
	'''
	normalize,feature selection and SMV model training
	'''
	dadosSample,labelSample=dataSampling(dados,label)

	'''
	normalize the data (with our implementation)
	'''
	print 'normalizing...'
	#normalize=NewMain().run(dadosSample,0)
	#normalize=maxMin_Normalizer().run()
	normalize=Normalizer().fit_transform(linhast)
	print 'done...'
	'''
	reduced with feature selection
	'''

	reduced,index=correlationCalculate(normalize)

	'''
	train SVM model
	'''	
	clf = svm.SVC()
	#clf = SGDClassifier(loss="hinge", penalty="l2")
	#clf = tree.DecisionTreeClassifier()
	model=clf.fit(reduced, labelSample)
	

	return model,index


train = open("classes-17-reduced.out", "r")

linhas=train.readlines()

sample,label=dataPreparing(linhas)

modelo,index=traininML(sample,label)

print 'modelo criado'

features={} #to see the evolution of the features

label=[]
dados =[]
window=0
tamanhoJanela=1000

acc={}
pre={}

data = open("classes-17-end.out", "r")
#linha = data.readline()

linhas = data.readlines()

linhast,labelt=dataPreparing(linhas)

#linhast,labelSample=dataSampling(linhast,labelt)

for i in range(0,len(linhast), tamanhoJanela): #
	janela = linhast[i:i+tamanhoJanela]	
	label= labelt[i:i+tamanhoJanela]	


# while linha !="":
# 	janela = []
# 	label=[]
# 	while linha !="" and len(janela) < tamanhoJanela:
# 		tmp1 = linha.strip("\n").split(",")[5:-2]#removing IPsrc,IPdst,portsrc,portdsc,proto,class
# 		tmp2 = []
# 		for i in tmp1:
# 			tmp2.append(float(i))
# 		janela.append(tmp2)
# 		tmpLabel=float(linha.strip("\n").split(",")[-1])
# 		if tmpLabel != 0.0:
# 			tmpLabel=1.0
# 		label.append(tmpLabel)
# 		linha = data.readline() #creating the window

	#normalize=NewMain().run(janela,0)
	#normalize=maxMin_Normalizer().run()
	reduced=MatrixReducer(janela,index)
	classification=modelo.predict(reduced)
#	print classification_report(label,classification)
	acc[1]=accuracy_score(label,classification)#,normalize==True)
	pre[1]=average_precision_score(label,classification)#, average='binary')  
	print 'Ac :'+str(acc[1])
	print 'Pre :' +str(pre[1])

#	if window==0:
#		acc[0]=acc[1]

#	if ((acc[0]-acc[1]) < 0.1*acc[1]):	
#		modelo,index=traininML(janela,label)
#		reduced,index=correlationCalculate(normalize)
#		features[window]=index
	window+=1
	acc[0]=acc[1]
