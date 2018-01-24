import numpy as np
import scipy.stats as st
import math,sys,time
from sklearn.metrics import mean_squared_error

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
				#if (localMax[feature]+pivote > aux):
				bins.append([aux,localMax[feature]+pivote]) #adding the last value as max
				#else:
				#	bins.append([aux,aux+localMax[feature]]) #adding the last value as max
				binsTotal[feature]=bins
				bins=[]
			else:
				if (localMax[feature] > binsTotal[feature][-1][1]): 
					pivote=(binsTotal[feature][0][1]-binsTotal[feature][0][0])
					if pivote == 0:
						pivote = 1.0
					preMax=binsTotal[feature][-1][1]
					if (localMax[feature] - preMax > (pivote*numberBins)): ##isso aqui eh para acelerar senao fica muito gigante,
						pivote=pivote*numberBins 
					while (preMax <= localMax[feature]+pivote): #isso deveria mudar unicamente qdo os valores de max-min mudem
						binsTotal[feature].append([preMax,preMax+pivote])  #aqui tem error 
						preMax+=pivote
					binsTotal[feature].append([preMax,preMax+pivote])  #aqui tem error 
				if (localMin[feature] < binsTotal[feature][0][0]): 
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

def updateHistogram(janela,bins,new,values):
	'''
	this function is to update the previous histogram with the new values
	'''
	global numberBins
	global N
	aux={}
	for feature in range(N): #percorrer as colunas
		aux[feature] = {k: [] for k in range(len(bins[feature]))} #initialize dict of list
		for b in range(len(bins[feature])): #num of bins
			#	p=[x.astype(np.float64) for x in jan[:,i] if x.astype(np.float64) >= bins[i][j][0] and x.astype(np.float64) < bins[i][j][1]] #to see how many values we have in each bin
			var=0 
			for x in janela[:,feature]:
				if (x.astype(np.float64) >= bins[feature][b][0] and x.astype(np.float64) < bins[feature][b][1]):
					var+=1
					aux[feature][b].append(x.astype(np.float64))
			if b in values[feature].keys(): #That means that we increased the bins
				new[feature][b]=new[feature][b]+var
				values[feature][b]=values[feature][b]+aux[feature][b]
			else:
				new[feature][b]=var #so we need to add the new values
				values[feature][b]=aux[feature][b]
			var=0
			aux[feature][b]=[]
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
			aux[feature][bins]=(histogram[feature][bins]/float(numberSamples)) #numeros de amostras no bin
		
	return aux

def calculateZ(relative):
	'''
	'''
	global N
	Z={}
	for feature in range(N):
		Z[feature] = {k: [] for k in range(len(relative[feature]))} #initialize dict of list
		p=0
		for bins in relative[feature].keys():
			#p=filter(lambda x : x < relative[feature][bins], relative[feature].values()) #check the values of bins smaller than the current one
			#Z[feature][bins]=st.norm.cdf(sum(p))
			p+=relative[feature][bins]
			if p > 1.0:
				p=1.0
			norm=st.norm.ppf(p)
			if np.isinf(norm):
				if float('Inf') == norm:
					Z[feature][bins]=3.4
				if -float('Inf') == norm:
					Z[feature][bins]=-3.4
			else:
				Z[feature][bins]=norm
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


output=str(sys.argv[1])+'proposta'

output_file=open(output,'w')


global windowSize
windowSize=int(sys.argv[1]) #as paper
				

global windowsNumber #to see the number of the windows
windowsNumber = 0



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



time1=time.time()
batch=[]
a=lines[0:100000]
for i in a:
   batch.append(dataPrepare(i))

output_file.write(str('prepare time : '+str(time.time()-time1)+'\n'))

before=batch
batch=np.array(batch)

print 'file loaded'

histogram={}  #here are the histogram diveded in feature. Each feature has N bins

jan=[]  #take a windows everytime we have a batch

rawValues={} #dictionary of features with original values diveded by bins

relative={} #dictionary of features with relative frequency of each bins (frequency of the bin/totalSamples)

Zvalues={} #dictionary of features with Zvalues of each bins  (Z>P\left (x=\sum_{j}^{i} f_q_i \right ))

newValues={} #dictionary of features with maps between Zvalues and real values

final={} #this must be the final normalized result

beg=time.time()
for i in range(0,len(batch), windowSize): #
		
		jan = batch[i:i+windowSize]		
		
		localMax,localMin,localMean,localStd = getValues(jan)
		time2=time.time()
		output_file.write(str('getValues time : '+str(time2-time1)+'\n'))
		
		if windowsNumber==0:
			maxGlobal,minGlobal=localMax,localMin
			numberSamples= (N*windowSize)
			binsTotal=createBins(localMax,localMin)
			time3=time.time()
			output_file.write(str('createbins time : '+str(time3-time2)+'\n'))	
			histogram,rawValues=(createHistogram(jan,binsTotal))
			time6=time.time()
			output_file.write(str('createHistogram time : '+str(time6-time3)))
		else:
			binsTotal=updateBins(localMax,localMin,binsTotal)	
			time5=time.time()
			output_file.write(str('updateBins time : '+str(time5-time2)+'\n'))
			numberSamples=(N*windowSize*windowsNumber)
			histogram,rawValues=(updateHistogram(jan,binsTotal,histogram,rawValues))
			time6=time.time()
			output_file.write(str('updateHistogram time : '+str(time6-time5)+'\n'))


			
		relative=(relativeFreq(histogram,1000))
		time7=time.time()
		output_file.write(str('relativeFreq time : '+str(time7-time6)+'\n'))
		Zvalues=(calculateZ(relative))
		time8=time.time()
		output_file.write(str('Zvalues time : '+str(time8-time7)+'\n'))
		newValues=backZ2values(rawValues,Zvalues)
		time9=time.time()
		output_file.write(str('newValues time : '+str(time9-time8)+'\n'))
		t=return2dataset(jan,rawValues,newValues)
			

		if windowsNumber==0:
			final=t
		else:
			final=np.vstack((final,t))

		time10=time.time()
		output_file.write(str('window time : '+str(time10-time2)+'\n'))
	 	windowsNumber+=1 #incrementing this number
		if (windowsNumber % 1000) == 0:
			print "windowsNumber: "+str(windowsNumber)


end=time.time()-beg

original_maxmin=[]
final=np.asfarray(final)
t=np.asfarray(before)
for i in range(N):
	original_maxmin.append(mean_squared_error(t[:,i],final[:,i]))

output_file.write(str(original_maxmin)+'\n')

output_file.write(str('processing time : '+str(end)))

output_file.close()

# t=0
# for i in range(len(histogram)):
# 	for j in histogram[i]:
# 		#if (histogram[i].values().count(0) > 7):
# 		#	f=1000
# 		#else:
# 		f=sum(histogram[i].values())
# 	t+=f
# 	if f != 1000:
# 		print i
# 		print f



