'''
Machine Learning Models
'''
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import csv
import numpy as np
from newmain import NewMain 
from maxMin_Normalizer import maxMin_Normalizer
from NewmaxMin import *


'''
normalizer do sklearner
'''
from sklearn.preprocessing import Normalizer


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

#to balance training dataset
from imblearn.over_sampling import SMOTE 
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

	#sm = SMOTE(ratio='minority') #to sample data
	sm = ADASYN(ratio='minority')
	dadosSample,labelSample=sm.fit_sample(dados,label)

	return dadosSample,labelSample

def traininML(dados,label,flag=0):
	'''
	normalize,feature selection and SMV model training
	'''

	if (flag==0):
		dadosSample,labelSample=dataSampling(dados,label)

		'''
		normalize the data (with our implementation)
		'''

		
		print 'normalizing...'
		normalize=NewMain().run(dadosSample,0)
		#normalize=maxMin_Normalizer().run()
		#normalize=Normalizer().fit_transform(dadosSample)
		#normalize=norm.run(dadosSample,0,0)

		print 'done...'
		'''
		reduced with feature selection
		'''
		reduced,index=correlationCalculate(normalize)
		
		data2train=reduced
		label=labelSample
	else:
		data2train,index=correlationCalculate(dados)

	'''
	train SVM model
	'''	
	#clf = svm.SVC(kernel='rbf')
	#clf = GaussianNB()
	#clf = SGDClassifier(loss="hinge", penalty="l2")
	clf = tree.DecisionTreeClassifier()
	model=clf.fit(data2train, label)
	

	return model,index

norm=NewmaxMin()
train = open("classes-17-reduced.out", "r")

linhas=train.readlines()

sample,label=dataPreparing(linhas)

modelo,index=traininML(sample,label)



print 'modelo criado'

features={} #to see the evolution of the features

features[0]=index

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

#dadosSample,labelSample=dataSampling(linhast,labelt)

#linhast,labelSample=dataSampling(linhast,labelt)

metrics={}

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
	
	reduced=MatrixReducer(janela,index)
	normalize=NewMain().run(reduced,0)
	#normalize=maxMin_Normalizer().run()

	#normalize=norm.run(janela,0,window+1)
	
	classification=modelo.predict(normalize)
#	print classification_report(label,classification)
	acc[1]=accuracy_score(label,classification)#,normalize==True)
	pre[1]=average_precision_score(label,classification)#, average='binary')  
	print 'Ac :'+str(acc[1])
	print 'Pre :' +str(pre[1])
	
	metrics[window]=acc[1],pre[1]

	if window==0:
		acc[0]=acc[1]

	if (acc[0]>acc[1]) and (acc[0]-acc[1] !=0):
		if (abs(acc[0]-acc[1]) >= 0.05) or acc[1] <0.9:#*acc[1]) :
			print	'probable concept drift'
			print window
			modelo,index=traininML(janela,label,1)
			#reduced,index=correlationCalculate(normalize)
			features[window]=index
	window+=1
	acc[0]=acc[1]



'''
to save in file


output2=open('salidaTree.csv','w')

for i in range(len(metrics)):

	output2.write(str(i+1)+','+str(metrics[i][0])+','+str(metrics[i][1])+'\n')
output2.close()

'''