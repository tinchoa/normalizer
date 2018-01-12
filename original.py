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
		for b in range(int(numberBins)+1): #num of bins
			p=[x.astype(np.float64) for x in janela[:,feature] if x.astype(np.float64) >= bins[feature][b][0] and x.astype(np.float64) < bins[feature][b][1]] #to see how many values we have in each bin
			new[feature][b]=len(p)
			values[feature][b]=(p)
			#aux2.append(p)
			#aux.append(len(p))
		#new[i]={j:aux}
		#values[i]={j:aux2}
		#aux=[]
		#aux2=[]

	return new,values


def return2dataset(janela,rawValues,newValues):
	'''
	this function will map the original values in the dataset with the values of the Z
	'''
	for feature in range(len(janela[0])): #percorrer as colunas
		for bins in range(len(rawValues[feature])): #num of bins
			for x in range(len(janela[:,feature])):
				if float(janela[x,feature]) in rawValues[feature][bins]:
						janela[x,feature] = newValues[feature][bins][0]
	return janela


				

global windowsNumber #to see the number of the windows
windowsNumber = 0


global windowSize 
windowSize=50

global N
N=39 #number of features

global numberSamples
numberSamples=1

global numberBins
#numberBins=math.ceil(math.sqrt(N))

numberBins=10

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
a=lines[0:10000]#1000]
for i in a:
   batch.append(dataPrepare(i))

before=batch
batch=np.array(batch)

histogram={}  #here are the histogram diveded in feature. Each feature has N bins

jan=[]  #take a windows everytime we have a batch

rawValues={} #dictionary of features with original values diveded by bins

relative={} #dictionary of features with relative frequency of each bins (frequency of the bin/totalSamples)

Zvalues={} #dictionary of features with Zvalues of each bins  (Z>P\left (x=\sum_{j}^{i} f_q_i \right ))

newValues={} #dictionary of features with maps between Zvalues and real values

final={} #this must be the final normalized result



localMax,localMin,localMean,localStd = getValues(batch)
		
binsTotal=createBins(localMax,localMin)
histogram,rawValues=(createHistogram(batch,binsTotal))




