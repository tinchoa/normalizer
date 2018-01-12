import numpy as np
import scipy.stats as st

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

#def calGlobal(localMax,localMin,localMean,localStd): 
	'''here we will update the value of the local and calculate the global
	   the values of alpha are to give more importance to the new batches without forget the old values
	'''
	global globalMax #final values, after calculate the equ. this is gonna be a 1*n (n==len(batch[0]))
	global globalMin 
	global globalMean
	global globalStd 
	global windowsNumber #only to keep the value of the windows, could be deleted
	global janMax  #windows with averages values
	global janMin  
	global janMean 
	global janStd  
	
	alpha1 = 0.9

	n = len(janMax) # should be # number of elements to take into the mean windows
	

	if n < 5: #first case, we dont have anything
	 	if n == 0:
			janMax.append(localMax)
 			globalMax.append(localMax)
 			janMin.append(localMin)
 			globalMin.append(localMin)
	 	else:
	 		janMax.append(localMax)
	 		janMin.append(localMin)

	else: #to keep only 5 values in the window
		janMax.pop(0)
		janMax.append(localMax)
		janMin.pop(0)
		janMin.append(localMin)
	
	auxMax=np.array(janMax)
	auxMin=np.array(janMin)

	auxTestMax=[]
	auxTestMin=[]
	
	for i in range(39):
		janelaMediaMax= float(sum(auxMax[:,i][0:len(auxMax)])/float(len(auxMax))) # sum of a from 0 index to 9 index. sum(a) == sum(a[0:len(a)]
		janelaMediaMin= float(sum(auxMin[:,i][0:len(auxMin)])/float(len(auxMin))) # sum of a from 0 index to 9 index. sum(a) == sum(a[0:len(a)]

		auxTestMax.append((janelaMediaMax*alpha1)+((1-alpha1)*localMax[i]))
		auxTestMin.append((janelaMediaMin*alpha1)+((1-alpha1)*localMin[i]))

	globalMax[0]=auxTestMax
	globalMin[0]=auxTestMin
	
def normalizing(janela,refMax,refMin):
	# normalized = (x-min(x))/(max(x)-min(x))
	global N

	for i in range(N):
		aux=np.subtract(janela[:,i].astype(np.float64),refMin[i])
		aux2=np.subtract(refMax[i],refMin[i])
		if (aux2 == 0):
			janela[:,i]=0.5 #https://docs.tibco.com/pub/spotfire/7.0.0/doc/html/norm/norm_scale_between_0_and_1.htm
			#If Emax is equal to Emin then Normalized (ei) is set to 0.5.
		else:
			janela[:,i]=np.nan_to_num(np.divide(aux,aux2).tolist())

	return janela

def verifyMetrics(localMax,localMin,refMax,refMin):
	'''
	function to verify if the values of the current chunks are different that references. (procedure metrics in paper)
	'''
	global N
	global windowSize
	global m1 #metric1 treshold 
	global m2 #metric2 threshold
	metric1 = False
	metric2 = False
	metric1Counter = 0
	for i in range(N):
		if (localMin[i] < refMin[i]):
			metric1Counter+=1
		if (localMax[i] > refMax[i]):
			metric1Counter+=1
		if ((refMin[i]-localMin[i])/refMin[i] > m2):
			metric2 = True
		if ((localMax[i]-refMax[i])/refMax[i] > m2):
			metric2 = True
	if (metric1Counter/windowSize > m1):
		metric1 = True

	return metric1,metric2

	
#main



global windowSize
windowSize=250 #as paper

global N
N=39 #number of features


global m1 #metric1 treshold 
global m2 #metric2 threshold

m1=0.05
m2=0.05

numberBins=math.ceil(math.sqrt(N))

windowsNumber = 0


janMax  = [] #janela de valores medios. Vou manter N valores 
janMin  = []
janMean = []
janStd  = []

histogram = {} #histogram with frequency of the samples
for j in range(N):
	histogram[j]=0

files=open('classes-17.out','r')
lines=files.readlines()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})

batch=[]
a=lines[0:10000]
for i in lines:
   batch.append(dataPrepare(i))

before=batch
batch=np.array(batch)


alpha1 = 0.9

x=[]
#def janela(batch): #janela = [e[i:i+windowSize] for i in range(len(e))]
''' calculate the sliding windows batch and send to obtain the values'''
global windowsNumber #to see the number of the windows
#windowsNumber+=1 #incrementing this number
jan=[]  #take a windows everytime we have a batch
test=[]

for i in range(0,len(batch), windowSize): #
		jan = batch[i:i+windowSize]		
		#calGlobal(localMax,localMin,localMean,localStd)

		if windowsNumber == 0:
			refMax,refMin,localMean,localStd = getValues(jan)
			salida=normalizing(jan,refMax,refMin)
		else:
			localMax,localMin,localMean,localStd=getValues(jan)
			metric1,metric2=verifyMetrics(localMax,localMin,refMax,refMin)
			if (metric1 == True and metric2 == True):
				refMax=localMax
				refMin=localMin
			
			salida=normalizing(jan,refMax,refMin)
		
		windowsNumber+=1 #incrementing this number

	#	p=tabelaZ(histogram)


