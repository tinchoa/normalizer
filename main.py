from histograma import Histograma
from original import Original
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
		saida = open("classes-17-norm-histo.out", "w")
		tamanhoJanela = int(sys.argv[1]) #as paper
		
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
			
			resultados = []
	
			for i in range(len(janela)):
				resultados.append([])
				for j in range(len(janela[i])):
					resultados[-1].append(hists[j].getNormalizedValues(janela[i][j])[0])
	


			for k in resultados:
				tmp = []
				tmp2= []
		 		for l in k:
		 			tmp.append(str(l))
		 			tmp2.append(l)
		 		resFinal.append(tmp2)
		 		linhaSaida =  ",".join(tmp)
		 		saida.write(linhaSaida+"\n")
			
		end=time.time()-beg

		# saida.write(str('processing time : '+str(end))+'\n')
			


		return resFinal,end

		#print janela
	


if __name__ == "__main__":

	output_file=open(str(sys.argv[1])+'-output','w')
	print 'proposal starting... '+'\n'
	proposal,timeProposal=Main().run()
	print 'proposal finished... '+'\n'
	
	print 'original started...'+'\n'
	old=Original().run()	

	print 'maxMin started....'+'\n'
	maxMin,timeMaxmin=maxMin_Normalizer().run()
	
	'''
	to calculate the mean square error
	'''
	original_proposal=[]
	original_max=[]
	original=np.asfarray(old)
	proposal=np.asfarray(proposal)
	maxMin=np.asfarray(maxMin)
	for i in range(len(proposal[0])):
	 	original_proposal.append(mean_squared_error(original[:,i],proposal[:,i]))
	 	original_max.append(mean_squared_error(original[:,i],maxMin[:,i]))
	
	MSEproposal=mean_squared_error(original,proposal)
	MSEmaxMin=mean_squared_error(original,maxMin)


	output_file.write(str(MSEproposal)+','+str(timeProposal)+'\n')
	#output_file.write('Proposal Procesing time: '+ str(timeProposal)+'\n')
	output_file.write(str(MSEmaxMin)+','+str(timeMaxmin)+'\n')
	#output_file.write('MaxMin Procesing time: '+ str(timeMaxmin)+'\n')


	output_file.close()