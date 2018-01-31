import numpy as np
import scipy.stats as st
import math,time
import sys
from sklearn.metrics import mean_squared_error


class NewmaxMin:

	def __init__(self):
		#windowSize=int(sys.argv[1]) #as paper
		self.windowSize=1000
		self.N=40 #number of features

		# global m1 #metric1 treshold 
		# global m2 #metric2 threshold

		self.m1=0.05
		self.m2=0.05

		self.refMax=[]
		self.refMin=[]
		self.salida=0
		numberBins=math.ceil(math.sqrt(self.N))
		np.set_printoptions(precision=3)
		np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})

	#output=str(sys.argv[1])+'max-min'

	#output_file=open(output,'w')

	def getValues(self,janela):
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

	def normalizing(self,janela):
		# normalized = (x-min(x))/(max(x)-min(x))

		for i in range(len(self.refMin)):
			aux=np.subtract(janela[:,i].astype(np.float64),self.refMin[i])
			aux2=np.subtract(self.refMax[i],self.refMin[i])
			if (aux2 == 0):
				janela[:,i]=0.5 #https://docs.tibco.com/pub/spotfire/7.0.0/doc/html/norm/norm_scale_between_0_and_1.htm
				#If Emax is equal to Emin then Normalized (ei) is set to 0.5.
			else:
				janela[:,i]=np.nan_to_num(np.divide(aux,aux2).tolist())
			for j in range(len(janela[:,i])):
				if float(janela[:,i][j])>1:
					janela[:,i][j]=1


		return janela

	def verifyMetrics(self,localMax,localMin):
		'''
		function to verify if the values of the current chunks are different that references. (procedure metrics in paper)
		'''
		metric1 = False
		metric2 = False
		metric1Counter = 0
		for i in range(len(localMin)):
			if (localMin[i] < self.refMin[i]):
				metric1Counter+=1
			if (localMax[i] > self.refMax[i]):
				metric1Counter+=1
			if self.refMin[i] == 0: #to avoid zero division
				if ((self.refMin[i]-localMin[i])/1 > self.m2):
					metric2 = True
			else:
				if ((self.refMin[i]-localMin[i])/self.refMin[i] > self.m2):
					metric2 = True
			if self. refMax[i]==0:#to avoid zero division
				if ((localMax[i]-self.refMax[i])/1 > self.m2):
					metric2 = True
			else:
				if ((localMax[i]-self.refMax[i])/self.refMax[i] >self. m2):
					metric2 = True
		if (metric1Counter/self.windowSize > self.m1):
			metric1 = True

		return metric1,metric2


		
		end=time.time()-beg




		''' to write in file'''
		# for k in salida:
		# 	tmp = []
		# 	for l in k:
		# 		tmp.append(str(l))
		# 	linhaSaida =  ",".join(tmp)
		# 	saida.write(linhaSaida+"\n")

		# end=time.time()-beg

		# saida.write(str('processing time : '+str(end)))

		# saida.close()


		'''
		to calculate the mean square error
		'''
		# original_maxmin=[]
		# t=np.asfarray(before)
		# salida=np.asfarray(salida)
		# for i in range(N):
		# 	original_maxmin.append(mean_squared_error(t[:,i],salida[:,i]))

		# output_file.write(str(original_maxmin)+'\n')

		# output_file.write(str('processing time : '+str(end)))

		# output_file.close()

	def run(self,data,flag,windowsNumber):
		beg=time.time()

		if flag==1: #working normal with file upload
			windowsNumber = 0
			files=open('classes-17.out','r')
			saida = open("max-min-classes-17-norm.out", "w")

			lines=files.readlines()

			batch=[]
			#a=lines[0:100000]
			for i in lines:
			   batch.append(dataPrepare(i))

			before=batch
			batch=np.array(batch)

			print 'file loaded'

			jan=[]  #take a windows everytime we have a batch

			beg=time.time()
			for i in range(0,len(batch), windowSize): #
					jan = batch[i:i+windowSize]		
					#calGlobal(localMax,localMin,localMean,localStd)

					if windowsNumber == 0:
						refMax,refMin,localMean,localStd = getValues(jan)
						salida=normalizing(jan,refMax,refMin)
					else:
						localMax,localMin,localMean,localStd=getValues(jan)
						metric1,metric2=verifyMetrics(localMax,localMin,refMax,refMin)
						if (metric1 and metric2):
							refMax=localMax
							refMin=localMin
						
						t=normalizing(jan,refMax,refMin)		

					if windowsNumber!=0:
						salida=np.vstack((salida,t))
					
					windowsNumber+=1 #incrementing this number
					if (windowsNumber % 1000) == 0:
						print "windowsNumber: "+str(windowsNumber)

			#	lower, upper = 0, 1
			#	salida=np.asfarray(salida)
			#	salidaNew = [lower + (upper - lower) * x for x in salida]
		
			#output=str(sys.argv[1])+'max-min'
			def dataPrepare(item):
				''' get the values, remove the categorical data'''
				a=item.split(',')
				label=a[len(a)-1].split('\n')[0]
				data=a[5:len(a)-1]#removing IPsrc,IPdst,portsrc,portdsc,proto
				return data

		if flag==0:
			jan=np.array(data)
			if windowsNumber == 0:
				self.refMax,self.refMin,localMean,localStd = self.getValues(jan)
				self.salida=self.normalizing(jan)
			else:
				localMax,localMin,localMean,localStd=self.getValues(jan)
				metric1,metric2=self.verifyMetrics(localMax,localMin)
				if (metric1 and metric2):
					refMax=localMax
					refMin=localMin
			self.salida=self.normalizing(jan)		
#			if windowsNumber!=0:
#				self.salida=np.vstack((self.salida,t))

		return self.salida.tolist()#,end
