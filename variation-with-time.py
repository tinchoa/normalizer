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


Machine Learning Models
'''
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
		#normalize=NewMain().run(dadosSample,0)
		#normalize=maxMin_Normalizer().run()
		normalize=Normalizer().fit_transform(dadosSample)
		#normalize=norm.run(dadosSample,0,0)

		print 'done...'
		'''
		reduced with feature selection
		'''

		grupo={}
		grupo[1]=range(0,4)
		grupo[2]=range(4,12)
		grupo[3]=range(12,20)
		grupo[4]=range(20,29)
		grupo[5]=range(29,33)
		grupo[6]=range(33,37)
		grupo[7]=range(37,39)

		# grupo1=range(0,4)
		# grupo2=range(4,12)
		# grupo3=range(12,20)
		# grupo4=range(20,29)
		# grupo5=range(29,33)
		# grupo6=range(33,37)
		# grupo7=range(37,40)

		index=grupo[int(sys.argv[1])]
		#index=grupo[1]
		reduced=MatrixReducer(normalize,index)

		#reduced,index=correlationCalculate(normalize)
		
		data2train=reduced
		label=labelSample
	else:
		data2train,index=correlationCalculate(dados)

	'''
	train SVM model
	'''	
	if int(sys.argv[2]) == 1:
		clf = KNeighborsClassifier(n_neighbors=3)
		classifier='KNN'
	if int(sys.argv[2]) == 2:
		clf = MLPClassifier(alpha=1, random_state=1)
		classifier='MLP'
	if int(sys.argv[2]) == 3:
		clf = RandomForestClassifier(max_depth=2, random_state=0)
		classifier='RF'
	if int(sys.argv[2]) == 4:
		clf = svm.SVC(kernel='rbf')
		classifier='SVM-RBF'
	if int(sys.argv[2]) == 5:
		clf = svm.SVC(kernel='linear')
		classifier='SVM-RBF'
	if int(sys.argv[2]) == 6:
		clf = GaussianNB()
		classifier='GNB'
	if int(sys.argv[2]) == 7:
		clf = SGDClassifier(loss="hinge", penalty="l2")
		classifier='SDG'
	if int(sys.argv[2]) == 8:
		clf = tree.DecisionTreeClassifier(random_state=0)
		classifier='Tree'
	model=clf.fit(data2train, label)
	

	return model,index,classifier

norm=NewmaxMin()
train = open("classes-17-reduced.out", "r")

linhas=train.readlines()

sample,label=dataPreparing(linhas)

modelo,index,classifier=traininML(sample,label)



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
	
	
	normalize=NewMain().run(janela,0)
	#normalize=maxMin_Normalizer().run()

	#normalize=norm.run(janela,0,window+1)
	reduced=MatrixReducer(normalize,index)
	classification=modelo.predict(reduced)
#	print classification_report(label,classification)
	acc[1]=accuracy_score(label,classification)#,normalize==True)
	pre[1]=average_precision_score(label,classification)#, average='binary')  
	if len(set(label)) == 2: 
		tn, fp, fn, tp = confusion_matrix(label,classification).ravel()
		recall=recall_score(label,classification, average='binary')
	else: 
		tn,fp,fn,tp=0,0,0,0
		recall=0
	print 'Ac :'+str(acc[1])
	print 'Pre :' +str(pre[1])
	print 'FP :' +str(fp)
	print 'FN :' +str(fn)
	print 'TP :' +str(tp)
	print 'TN :' +str(tn)
	print 'Recall: '+str(recall)

	metrics[window]=acc[1],pre[1],tn, fp, fn, tp,recall

	# if window==0:
	# 	acc[0]=acc[1]

	# if (acc[0]>acc[1]) and (acc[0]-acc[1] !=0):
	# 	if (abs(acc[0]-acc[1]) >= 0.05): #or acc[1] <0.9:#*acc[1]) :
	# 		print	'probable concept drift'
	# 		print window
	# 		modelo,index=traininML(janela,label,1)
	# 		#reduced,index=correlationCalculate(normalize)
	# 		features[window]=index
	window+=1
	acc[0]=acc[1]



'''
to save in file
'''

output2=open('grupos/'+'grupo'+str(int(sys.argv[1]))+'-'+classifier+'-nossa-1000.csv','w')

for i in range(len(metrics)):

	output2.write(str(i+1)+','+str(metrics[i][0])+','+str(metrics[i][1])+'\n')

tmp=0
for i in metrics:
	tmp+=metrics[i][0]
avg=tmp/float(len(metrics))

print avg

output2.write(str(avg)+'\n')

output2.close()

