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
		pivote=(localMax[i]-localMin[i])/numberBins
		aux=localMin[i]
		for j in range(int(numberBins)):
			bins.append([aux,aux+pivote])
			aux+=pivote
			x = [k for k in janela[:,k] if i>aux and i<aux+pivote]
			frequency[j]=x #ver si esto esta funcionando
			x=0		
		columns[i]=bins
		bins=[]

	return columns


def createHistogram(janela,columns):
	'''
	aqui devo percorrer os valores das amostras e ver em que bin cai
	con eso devo crear nuevos bins con los valores
	'''

	###como esta ahora esta errado, deveria crear nuevos bins, una estructura similar a la colums
	new=[]
	flag=0
	for i in range(N): #columns
		col=[]
		for j in range(windowSize): #number of samples
			sample=janela[:,i][j]
			for k in range(int(numberBins)): #number of bins
				bind=[]
				if (columns[i][k][0] >= sample.astype(np.float64) <= columns[i][k][1]): # and flag ==0:
					bind.append(sample.astype(np.float64))
					#flag=1
			col.append(bind)
		new.append(col)
			#if  flag == 0: # that's mean the value didn't enter in any bin 


	return new





global windowSize
windowSize=50

global N
N=39 #number of features

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

columns={}


for i in range(0,len(batch), windowSize): #
		windowsNumber+=1 #incrementing this number
		
		jan = batch[i:i+windowSize]		
		
		localMax,localMin,localMean,localStd = getValues(jan)
		
		columns[windowsNumber]=createBins(localMax,localMin)

		x.append(createHistogram(jan,columns[windowsNumber]))
