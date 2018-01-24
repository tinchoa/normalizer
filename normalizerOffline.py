import numpy as np
import scipy.stats as st
import math,time
from sklearn import preprocessing

def normalizerOffline():

	w = open("/tmp/classes-17-norm-offline.out",'w')
	f = open("classes-17.out",'r')
	linha = f.readline()
	while linha != "":
		vetor = [float(i) for i in linha.split(",")[5:-1]]
		for i in range(len(vetor)):
			try:
				vetor[i] = (vetor[i]-vetorMin[i])/(vetorMax[i]-vetorMin[i])
			except ZeroDivisionError:
				if (vetorMin[i]+vetorMax[i]) > 0: vetor[i] = (vetor[i])/(vetorMax[i]+vetorMin[i])
				else: vetor[i] = 0.5
		linha = f.readline()
		vetorStr = [str(i) for i in vetor]
		w.write(",".join(vetorStr)+"\n")



def maxMin():
	f = open("classes-17.out",'r')
	linha = f.readline()
	vetor = [float(i) for i in linha.split(",")[5:-1]]
	vetor
	vetorMax = vetor
	vetorMin = [float(i) for i in linha.split(",")[5:-1]]
	while linha != "":
		vetor = [float(i) for i in linha.split(",")[5:-1]]
		for i in range(len(vetor)):
			if vetor[i] > vetorMax[i]:
				vetorMax[i] = vetor[i]
			if vetor[i] < vetorMin[i]:
				vetorMin[i] = vetor[i]
		linha = f.readline()
	vetorMax
	vetorMin

	