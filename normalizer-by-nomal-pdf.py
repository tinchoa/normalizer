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
		vmax.append(max(column))
		vmin.append(min(column))
		umean.append(np.mean(column))
		sigmin.append(np.std(column)) 
	return vmax,vmin,umean,sigmin


def createBins(localMax,localMin,janela):
	'''
	function to create the limits of each bin. Each column is composed of numberBins=math.ceil(math.sqrt(N))
	we will have 2 list, one with the max-min of each bean, and one withe the values of the bins

	'''
	global numberBins

	bins=[]
	columns={}
	frequency={}
	for i in range(N):
		pivote=(localMax[i]-localMin[i])/(numberBins)
		pivote=np.ceil(pivote) #to round
		aux=localMin[i]
		for j in range(int(numberBins)):
			bins.append([aux,aux+pivote])
			aux+=pivote
		bins.append([aux,localMax[i]+pivote]) #adding the last value as max
		columns[i]=bins
		bins=[]

	return columns 


def createHistogram(janela,bins):
	'''
	aqui devo percorrer os valores das amostras e ver em que bin cai
	con eso devo crear nuevos bins con los valores
	'''
	global numberBins
	global N
	new={}
	aux=[]
	for i in range(N): #percorrer as colunas
		new[i]=[]
		for j in range(int(numberBins)+1): #num of bins
				p=[x.astype(np.float64) for x in jan[:,i] if x.astype(np.float64) >= bins[i][j][0] and x.astype(np.float64) < bins[i][j][1]] #to see how many values we have in each bin
				aux.append(len(p))
				# var=0 #ver eso aqui
				# for x in janela[:,i]:
				# 	if (x.astype(np.float64) >= columns[i][j][0] and x.astype(np.float64) < columns[i][j][1]):
				#  		var+=1
				#  		flag=1
				#  	else:
				#  		flag=0
				# if flag==0:
				# 	g+=1
				# aux.append(var)
				# var=0
			#p=[x.astype(np.float64) for x in janela[:,i] if x.astype(np.float64) >= columns[i][j][0]  and x.astype(np.float64) < columns[i][j][1]] #to see how many values we have in each bin
		new[i]=aux
		aux=[]

	return new





global windowSize
windowSize=50

global N
N=39 #number of features

global numberBins
numberBins=math.ceil(math.sqrt(N))

windowsNumber = 0

janMax  = [] #janela de valores medios. Vou manter N valores 
janMin  = []

histogram = {} #histogram with frequency of the samples
for j in range(N):
	histogram[j]=0

files=open('classes-17.out','r')
lines=files.readlines()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})

batch=[]
a=lines[0:1000]
for i in a:
   batch.append(dataPrepare(i))

before=batch
batch=np.array(batch)

x=[]
#def janela(batch): #janela = [e[i:i+windowSize] for i in range(len(e))]
''' calculate the sliding windows batch and send to obtain the values'''
global windowsNumber #to see the number of the windows
#windowsNumber+=1 #incrementing this number
jan=[]  #take a windows everytime we have a batch

frequency={}
binsTotal={}

b=[]

for i in range(0,len(batch), windowSize): #
		
		jan = batch[i:i+windowSize]		
		
		localMax,localMin,localMean,localStd = getValues(jan)
		
		binsTotal[windowsNumber]=createBins(localMax,localMin,jan)
		#,frequency[windowsNumber]

		x.append(createHistogram(jan,binsTotal[windowsNumber]))
	
		windowsNumber+=1 #incrementing this number



t=0
for i in range(len(x)):
	for j in x[i]:
		if (x[i][j].count(0)==numberBins+1):
			f=50
		else:
			f=sum(x[i][j])
		t+=f
		if f != 50:
			print i,j
			print f

