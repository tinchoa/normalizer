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


def calGlobal(localMax,localMin,localMean,localStd): 
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
	
def normalizing(janela):
	# normalized = (x-min(x))/(max(x)-min(x))

	global globalMax 
	global globalMin 
	global globalMean
	global globalStd 

	for i in range(39):
		aux=np.subtract(janela[:,i].astype(np.float64),globalMin[0][i])
		aux2=np.subtract(globalMax[0][i],globalMin[0][i])
		if (aux2 == 0):
			janela[:,i]=0.5 #https://docs.tibco.com/pub/spotfire/7.0.0/doc/html/norm/norm_scale_between_0_and_1.htm
			#If Emax is equal to Emin then Normalized (ei) is set to 0.5.
		else:
			janela[:,i]=np.nan_to_num(np.divide(aux,aux2).tolist())

	return janela


def updateHisto(histo,janela):
	'''
	function to update the values of the histogram we are mantaining
	'''
	global windowSize

	for j in range(39):
		aux=np.sum(janela[:,j].astype(np.float64))
		aux2=aux+histo[j]
		histo[j]=np.nan_to_num(np.divide(aux2,windowSize))

	return histo
		
def tabelaZ(histo):
	'''
	get the values of the histogram and pass them to a Z table value
	'''
	
	new={}
	aux=0
	for i in range(len(histogram)):
		for j in range(len(histogram)):
			if histogram[i] > histogram[j]: #check the values with minor relative frequency 
				aux+=histogram[j] #then sum
			else:
				aux=histogram[i]
		print aux
		new[i]=st.norm.ppf(aux) ##Z table with mean 0 std 1
		aux=0

	j=np.histogram(new.values(),bins=6)

	return test



#main



global windowSize
windowSize=30

global N
N=39 #number of features

numberBins=math.ceil(math.sqrt(N))


windowsNumber = 0
globalMax  = []
globalMin  = []
globalMean = []
globalStd  = []

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
		windowsNumber+=1 #incrementing this number
		jan = batch[i:i+windowSize]		
		test=jan
		localMax,localMin,localMean,localStd = getValues(jan)
		#calGlobal(localMax,localMin,localMean,localStd)

		
		#x.append(normalizing(test))
		for j in range(N):
			aux=np.subtract(jan[:,j].astype(np.float64),localMin[j])
			aux2=np.subtract(localMax[j],localMin[j])
			jan[:,j]=np.divide(aux,aux2)	

		histogram=updateHisto(histogram,test)
		p=tabelaZ(histogram)


