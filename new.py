import numpy as np
import scipy.stats as st
import math

def dataPrepare(item):
	''' get the values, remove the categorical data'''
	a=item.split(',')
	label=a[len(a)-1].split('\n')[0]
	data=a[5:len(a)-2]#removing IPsrc,IPdst,portsrc,portdsc,proto
	return data


def getValues(janela):
	''' take the local values of the current batch'''
	vmax=[]
	vmin=[]
	umean=[]
	sigmin=[]    
	for i in range(len(janela[0])):
		column=janela[:,i].astype(np.float64)
		localMax=max(column)
		vmax.append(localMax)
		localMin=min(column)
		vmin.append(localMin)
		umean.append(np.mean(column))
		sigmin.append(np.std(column)) 
		
	return vmax,vmin,umean,sigmin


def createBins(localMax,localMin):
	'''
	function to create the limits of each bin. Each column is composed of numberBins=math.ceil(math.sqrt(N))
	we will have 2 list, one with the max-min of each bean, and one withe the values of the bins

	'''
	global numberBins

	bins=[]
	columns={}
	frequency={}
	for i in range(N):
		if (localMin[i] == localMax[i]): #otherwise it's gonna be always zero
			pivote=1
		else:
			pivote=(localMax[i]-localMin[i])/(numberBins)
		pivote=np.ceil(pivote) #to round
		aux=localMin[i]
		for j in range(int(numberBins)):
			bins.append([aux,aux+pivote])
			aux+=pivote
		if localMax[i]+pivote > aux:
			bins.append([aux,localMax[i]+pivote]) #adding the last value as max
		else:
			bins.append([aux,aux+localMax[i]]) #adding the last value as max
		columns[i]=bins
		bins=[]

	return columns 


def updateBins(localMax,localMin,binsTotal):
	'''
	this function will update the boundaries of the bins of each column, in case it is lower/max than the current values 
	we will create new bins
	'''

	global numberBins
	for feature in binsTotal:
		for bins in range(len(binsTotal[feature])):
			if (binsTotal[feature].count(binsTotal[feature][0]) > 1): #check if all bins are the same, if not we will use the procedure of createbins
				bins=[] 
				if (localMin[i] == localMax[i]): #otherwise it's gonna be always zero
					pivote=1
				else:
					pivote=(localMax[i]-localMin[i])/(numberBins)
				pivote=np.ceil(pivote) #to round
				aux=localMin[feature]
				bins.append([aux,aux+pivote])
				aux+=pivote
				if (localMax[feature]+pivote > aux):
					bins.append([aux,localMax[feature]+pivote]) #adding the last value as max
				else:
					bins.append([aux,aux+localMax[feature]]) #adding the last value as max
				binsTotal[feature]=bins
				bins=[]
			else:
				if localMax[feature] > binsTotal[feature][len(binsTotal[feature])-1][1]:
				#print "max passed"
					pivote=(binsTotal[feature][0][1]-binsTotal[feature][0][0])
					if pivote == 0:
						pivote =1.0
					preMax=binsTotal[feature][len(binsTotal[feature])-1][1]
					if (localMax[feature] - preMax > 500 ): ##isso aqui é para acelerar senao fica muito gigante,
						pivote=500 #o que poderia ser mais inteligente?
					while (preMax <= localMax[feature]): #isso deveria mudar unicamente qdo os valores de max-min mudem
						binsTotal[feature].append([preMax,preMax+pivote])  #aqui hay un error 
						preMax+=pivote
				if localMin[feature] < binsTotal[feature][0][0]:
				#"print min passed"
					binsTotal[feature].insert(0,[localMin[feature],binsTotal[feature][0][0]])   

	return binsTotal 		


def createHistogram(janela,bins):
	'''
	function to create the histogram with the first values we have
	'''
	global numberBins
	global N
	new={}
	values={}
	aux=[]
	aux2=[]
	for feature in range(N): #percorrer as colunas
		values[feature] = {k: [] for k in range(int(numberBins)+1)} #initialize dict of list
		new[feature] = {k: 0 for k in range(int(numberBins)+1)} #initialize dict of list
		for j in range(int(numberBins)+1): #num of bins
			p=[x.astype(np.float64) for x in jan[:,feature] if x.astype(np.float64) >= bins[feature][j][0] and x.astype(np.float64) < bins[feature][j][1]] #to see how many values we have in each bin
			new[feature][j]=len(p)
			values[feature][j].append(p)
			#aux2.append(p)
			#aux.append(len(p))
		#new[i]={j:aux}
		#values[i]={j:aux2}
		#aux=[]
		#aux2=[]

	return new,values


def updateHistogram(janela,bins,new,values):
	'''
	this function is to update the previous histogram with the new values
	'''
	global numberBins
	global N
	aux={}
	for feature in range(N): #percorrer as colunas
		aux[feature] = {k: [] for k in range(len(bins[feature]))} #initialize dict of list
		for bind in range(len(bins[feature])): #num of bins
			#	p=[x.astype(np.float64) for x in jan[:,i] if x.astype(np.float64) >= bins[i][j][0] and x.astype(np.float64) < bins[i][j][1]] #to see how many values we have in each bin
			var=0 
			for x in janela[:,feature]:
				if (x.astype(np.float64) >= bins[feature][bind][0] and x.astype(np.float64) < bins[feature][bind][1]):
					var+=1
					aux[feature][bind].append(x.astype(np.float64))
			if bind in values[feature].keys(): #That means that we increased the bins
				new[feature][bind]=new[feature][bind]+var
				values[feature][bind]=values[feature][bind]+aux[feature][bind]
				
			else:
				new[feature][bind]=var #so we need to add the new values
				values[feature][bind]=aux[feature][bind]
			var=0
			aux[feature][bind]=[]
	return new,values


def relativeFreq(histogram, numberSamples):
	'''
	function to create the relative frequenci of each bin
	'''
	relative={}
	aux={}
	for feature in range(len(histogram)):
		aux[feature] = {k: [] for k in range(len(histogram[feature]))} #initialize dict of list

		for bins in range(len(histogram[feature])):
		#	print bins
			#aux.append(bins/float(numberSamples)) #isso aqui esta dando muito baixo (e acava caindo tudo perto de zero)
			aux[feature][bins]=(histogram[feature][bins]/float(50)) #numeros de amostras no bin
		#relative[feature][bins]=aux[feature][bins]
		#aux=[]
		#		histogram[feature][bins]
				#p=[x.astype(np.float64) for x in jan[:,i] if x.astype(np.float64) >= bins[i][j][0] and x.astype(np.float64) < bins[i][j][1]]

	return aux


def calculateZ(relative):
	'''
	'''
	global N
	Z={}
	for feature in range(N):
		Z[feature] = {k: [] for k in range(len(relative[feature]))} #initialize dict of list
		#for feature in relative[windows]:
		for bins in relative[feature]:
			if (relative[feature][bins] == 0.0):
				Z[feature][bins]=(st.norm.cdf(0))
			else:
				p=filter(lambda x : x < relative[feature][bins], relative[feature].values())
				Z[feature][bins]=st.norm.cdf(sum(p))
		
	return Z

def backZ2values(rawValues,Zvalues):
	'''
	this function will pass the Zvalues to the rawvalues 
	'''
	newValues={}
	for feature in Zvalues:
		newValues[feature] = {k: [] for k in range(len(relative[feature]))} #initialize dict of list
		for bins in Zvalues[feature]:
			newValues[feature][bins]=map(lambda x: Zvalues[feature][bins],rawValues[feature][bins])

	return newValues



global windowSize
windowSize=50

global N
N=39 #number of features

global numberSamples
numberSamples=1

global numberBins
numberBins=math.ceil(math.sqrt(N))

windowsNumber = 0

janMax  = [] #janela de valores medios. Vou manter N valores 
janMin  = []

# histogram = {} #histogram with frequency of the samples
# for j in range(N):
# 	histogram[j]=0

files=open('classes-17.out','r')
lines=files.readlines()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})

batch=[]
a=lines[0:1000]#1000]
for i in a:
   batch.append(dataPrepare(i))

before=batch
batch=np.array(batch)

histogram={}
#def janela(batch): #janela = [e[i:i+windowSize] for i in range(len(e))]
''' calculate the sliding windows batch and send to obtain the values'''
global windowsNumber #to see the number of the windows
#windowsNumber+=1 #incrementing this number
jan=[]  #take a windows everytime we have a batch

frequency={}
#binsTotal=

relative={}

rawValues={}

Zvalues={}

newValues={}

for i in range(0,len(batch), windowSize): #
		
		jan = batch[i:i+windowSize]		
		
		localMax,localMin,localMean,localStd = getValues(jan)
		
		if windowsNumber==0:
			maxGlobal,minGlobal=localMax,localMin
			numberSamples= (N*windowSize)
			binsTotal=createBins(localMax,localMin)
			histogram,rawValues=(createHistogram(jan,binsTotal))
		else:
			binsTotal=updateBins(localMax,localMin,binsTotal)	
			numberSamples=(N*windowSize*windowsNumber)
			histogram,rawValues=(updateHistogram(jan,binsTotal,histogram,rawValues))
			
			relative=(relativeFreq(histogram,numberSamples))

			Zvalues=(calculateZ(relative))

			newValues=backZ2values(rawValues,Zvalues)

	 	windowsNumber+=1 #incrementing this number



# t=0
# for i in range(len(histogram)):
# 	for j in histogram[i]:
# 		if (histogram[i].values().count(0) > 7):
# 			f=1000
# 		else:
# 			f=sum(histogram[i].values())
# 	t+=f
# 	if f != 1000:
# 		print i
# 		print f

