'''
run with 
python test-de-grupos.py <number of group> <classifier>

1:'KNN'
2:'MLP'
3:'RF'
4:'SVM-RBF'
5:'SVM-RBF'
6:'GNB'
7:'SDG'
8:'Tree'

'''

'''
ploting
'''
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

'''

Machine Learning Models
'''
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

##System
import csv
import numpy as np
import sys

#Normalizers
from newmain import NewMain 
from maxMin_Normalizer import maxMin_Normalizer
from NewmaxMin import *


'''
incremental 
'''
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

'''
normalizer do sklearner
'''
from sklearn.preprocessing import Normalizer


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

#to balance training dataset
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN 
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
	#sm = ADASYN(ratio='minority')
	dadosSample,labelSample=sm.fit_sample(dados,label)

	return dadosSample,labelSample

############################################
#################otro programa

data = open("classes-17-reduced.out", "r")
#linha = data.readline()

linhas = data.readlines()

linhast,labelt=dataPreparing(linhas)

dadosSample,labelSample=dataSampling(linhast,labelt)

#linhast,labelSample=dataSampling(linhast,labelt)

#tamanhoJanela=int(sys.argv[1])
tamanhoJanela=200
window=0
acc=[]
rec=[]
for i in range(0,len(dadosSample), tamanhoJanela):
	janela = dadosSample[i:i+tamanhoJanela]
	label= labelSample[i:i+tamanhoJanela]
	if window==0:
		 print window
		 clf = MultinomialNB()
		 normalize=Normalizer().fit_transform(janela)
		 reduced,index=correlationCalculate(normalize)
		 print index
		 model=clf.partial_fit(reduced,label,classes=np.unique(label))
	if window !=0:
		#print '-1: ' +str(label.tolist().count(-1))
		#print '+1: ' +str(label.tolist().count(1))
		normalize=NewMain().run(janela,0)

		#normalize=Normalizer().fit_transform(janela)
		#reduced,index=correlationCalculate(normalize)
		reduced=MatrixReducer(normalize,index)
		a=model.predict(reduced)
		acc.append(accuracy_score(label,a))
		#print a
		rec.append(recall_score(label,a, average='binary'))

#		model=clf.partial_fit(reduced,label)
	window+=1




data = open("classes-17-end.out", "r")

linhas = data.readlines()

linhast,labelt=dataPreparing(linhas)

for i in range(0,len(linhast), tamanhoJanela):
	janela = linhast[i:i+tamanhoJanela]
	label= labelt[i:i+tamanhoJanela]
	#normalize=Normalizer().fit_transform(janela)
	normalize=NewMain().run(janela,0)


	#reduced=MatrixReducer(normalize,index)
	reduced,index=correlationCalculate(normalize)
	print index

	a=model.predict(reduced)

	acc.append(accuracy_score(label,a))
		#print a
	rec.append(recall_score(label,a, average='binary'))
	model=clf.partial_fit(reduced,label)

#dadosSample,labelSample=dataSampling(linhast,labelt)

fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

ax.plot(acc,'b',rec,'r')
ax.set_xlabel('Window')
ax.set_ylabel('Accuracy')
path='/tmp/'
fig.savefig(path+'nex4.png')

plt.close(fig) 
plt.show()




# output2=open('incremental/'+'Bernoulli'+'-nossa-2000.csv','w')

# for i in range(len(acc)):

# 	output2.write(str(i+1)+','+str(acc[i])+','+str(rec[i])+'\n')

# output2.close()

# window=0
# b=[]    
# for i in range(0,len(dadosSample), tamanhoJanela):
# 	janela = dadosSample[i:i+tamanhoJanela]
# 	label= labelSample[i:i+tamanhoJanela]
# 	if window==0:
# 		 clf = MiniBatchKMeans(n_clusters=2, batch_size=tamanhoJanela, random_state=0)
# 		 model=clf.partial_fit(janela,label)#,classes=np.unique(label))
# 	if window !=0:
# 		a=model.predict(janela)
# 		acc=accuracy_score(label,a)#,normalize==True)
# 		#print a
# 		b.append(acc)
# 		model=clf.partial_fit(janela,label)
# 	window+=1
# import matplotlib.pyplot as plt

# plt.plot(b)
# plt.show()