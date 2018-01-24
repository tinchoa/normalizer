from histograma import Histograma
#from original import Original
from normalizerOffline import normalizer
from maxMin_Normalizer import maxMin_Normalizer

'''
added by Martin
'''
import time,sys
import numpy as np
from sklearn.metrics import mean_squared_error



####ate aqui

class Main:
	def run(self):
		beg=time.time()

		data = open("classes-17.out", "r")
#		saida = open("classes-17-norm-histo.out", "w")
		#tamanhoJanela = int(sys.argv[1]) #as paper
		tamanhoJanela=500
		self.hists = []
		hists = self.hists
		
		linha = data.readline()
		

		resFinal=[]
		while linha !="":
			janela = []
			while linha !="" and len(janela) < tamanhoJanela:
				tmp1 = linha.strip("\n").split(",")[5:-1]#removing IPsrc,IPdst,portsrc,portdsc,proto,class
				tmp2 = []
				for i in tmp1:
					tmp2.append(float(i))
				janela.append(tmp2)
				linha = data.readline()
			#processa janela
			#print len(janela)
			
			
			
			features = []
			
			for i in range(len(janela)):
				for j in range(len(janela[i])):
					if (len(features)-1) < j:
						features+=[[]]
					features[j].append(janela[i][j])
			
			
			for j in range(len(features)):
				if len(hists) < j+1:
					hists.append(Histograma(features[j]))
				else:
					hists[j].updateHistograma(features[j])
				#    print j, "hist", hists[j].hist, hists[j].pivo
					#print features[j]
			
	#	end=time.time()-beg			



	#	return hists,end

			resultados = []
	
			for i in range(len(janela)):
				resultados.append([])
				for j in range(len(janela[i])):
					resultados[-1].append(hists[j].getNormalizedValues(janela[i][j])[0])
	


	# 		for k in resultados:
	# 			tmp = []
	# 			tmp2= []
	# 	 		for l in k:
	# 	 			tmp.append(str(l))
	# 	 			tmp2.append(l)
	# 	 		resFinal.append(tmp2)
	# 	 		linhaSaida =  ",".join(tmp)
	# #	 		saida.write(linhaSaida+"\n")
			
		end=time.time()-beg

	# 	# saida.write(str('processing time : '+str(end))+'\n')
			


		##return hists,end #to retunr the object
		return resultados,end
	# 	#print janela
	


if __name__ == "__main__":

	# output_file=open(str(sys.argv[1])+'-output','w')
	print 'proposal starting... '+'\n'
	proposal,timeProposal=Main().run()
	print 'proposal finished... '+'\n'
	
	print 'original started...'+'\n'
	old=normalizer().run()	

	print 'maxMin started....'+'\n'
	maxMin,timeMaxmin=maxMin_Normalizer().run()
	
	'''
	to calculate the mean square error
	'''
	# original_proposal=[]
	# original_max=[]
	original=np.asfarray(old)
	proposal=np.asfarray(proposal)
	maxMin=np.asfarray(maxMin)
	for i in range(len(proposal[0])):
	  	original_proposal.append(mean_squared_error(original[:,i],proposal[:,i]))
	  	original_max.append(mean_squared_error(original[:,i],maxMin[:,i]))
	
	# #MSEproposal=mean_squared_error(original,proposal)
	# #MSEmaxMin=mean_squared_error(original,maxMin)

	MSEproposal=sum(original_proposal)/float(len(original_proposal))
	MSEmaxMin=sum(original_max)/float(len(original_max))

	# output_file.write(str(MSEproposal)+','+str(timeProposal)+'\n')
	# #output_file.write('Proposal Procesing time: '+ str(timeProposal)+'\n')
	# output_file.write(str(MSEmaxMin)+','+str(timeMaxmin)+'\n')
	# #output_file.write('MaxMin Procesing time: '+ str(timeMaxmin)+'\n')


	#output_file.close()