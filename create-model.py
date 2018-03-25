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
	normalize=Normalizer().fit_transform(dadosSample)
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

data = open("classes-25.out", "r")
#linha = data.readline()

linhas = data.readlines()

linhast,labelt=dataPreparing(linhas)

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
	reduced=MatrixReducer(janela,index)
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
	# print 'Ac :'+str(acc[1])
	# print 'Pre :' +str(pre[1])
	# print 'FP :' +str(fp)
	# print 'FN :' +str(fn)
	# print 'TP :' +str(tp)
	# print 'TN :' +str(tn)
	# print 'Recall: '+str(recall)

	metrics[window]=acc[1],pre[1],tn, fp, fn, tp,recall

	if window==0:
		acc[0]=acc[1]
	drift=[]
	if ((abs(acc[1]-acc[0])/acc[0]) >0.1) and (acc[0]-acc[1]) !=0:	
		modelo,index=traininML(janela,label)
		reduced,index=correlationCalculate(normalize)
		features[window]=index
		drift.append(window)
	window+=1
	acc[0]=acc[1]



output2=open('drift/'+'SVM-RBF-nosso-1000.csv','w')

for i in range(len(metrics)):

	output2.write(str(i+1)+','+str(metrics[i][0])+','+str(metrics[i][1])+'\n')

tmp=0
for i in metrics:
	tmp+=metrics[i][0]
avg=tmp/float(len(metrics))

print avg

output2.write(str(avg)+'\n')




output3=open('drift/'+'SVM-RBF-nosso-1000-drift.csv','w')

for i in range(len(features)):
	output3.write(str(i)+',')
	for j in range(len(features[i])):
		output3.write(+str(features[i])+',')
	output3.write(+str(drift[i][1])+'\n')


output2.close()
output3.close()

