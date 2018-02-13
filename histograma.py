import scipy.stats as st
import math
from copy import copy
import numpy


d = {}
for i in range (0,10000):
    k = float(i)/10000.
    d[k] = st.norm.ppf(k)
d[1] = numpy.Inf
    
def ppf(x):
    x = int(x*10000)/10000.0
    try:
        res = (d[x]+3.09)/(3.09*2)
        if res > 1:
            return 1
        else:
            if res < 0:
                res = 0
            return res
    except KeyError:
        return numpy.NAN


class Histograma:
    def __init__(self, valores=[], bins=0):
        self.total = 0
        self.createHistograma(valores, bins)
    
    def createHistograma(self, valores=[], bins=0):
        #print "valores", valores 
        if bins == 0:
            self.bins = math.sqrt(len(valores))
        if len(valores) == 0:
            self.valores = []
        else:
            self.valores = copy(valores)
        
                
        self.valores.sort()
        min = float(self.valores[0])
        #print "min", min
        max = float(self.valores[-1])
        #print "max", max
        
        if min == max:
            self.bins = 1
            pivo = 1
        else:        
            pivo = (max-min)/(self.bins-1)
            
        self.pivo = pivo
        #print self.pivo, max, min
        
        self.hist = []
        
        tmp = min
        i = 0 
        while len(self.hist) < self.bins:
            self.hist.append([tmp,tmp+pivo,0])
            for i in range(i, len(valores)):
                if self.valores[i] >= tmp and self.valores[i] < tmp+pivo:
                    self.hist[-1][2] += 1
                    self.total += 1
                else:
                    break
            tmp+= pivo
    
    def updateHistograma(self, valores):
        self.valores = copy(valores)
            
        self.valores.sort()
        
        if self.valores[0] < self.hist[0][0]:
            min = self.valores[0]
        else:
            min = self.hist[0][0]
        if self.valores[-1] > self.hist[-1][1]:
            max = self.valores[-1]
        else:
            max = self.hist[-1][1]
        
        self.bins = math.sqrt(self.total)
        
        if min == max:
            self.bins = 1
            pivo = 1
        else:        
            pivo = float(max-min)/float(self.bins-1)
            
        self.pivo = pivo
        
        while min < self.hist[0][0]:
            self.hist = [[self.hist[0][0]-self.pivo,self.hist[0][0],0]]+self.hist
        while max >= self.hist[-1][1]:
            self.hist = self.hist + [[self.hist[-1][1], self.hist[-1][1]+self.pivo, 0]]
        
        i = 0
        b = 0
        while b < len(self.hist):
            tmp = self.hist[b]
            for i in range(i, len(self.valores)):
                if self.valores[i] >= tmp[0]:
                    if self.valores[i] < tmp[1]:
                        tmp[2] += 1
                        self.total += 1
                else:
                    break
            b +=1

    def visualizeHistograma(self):

        return self.hist

            
    def normalizedHistograma (self):
        normalizedHist = []
        k = 0
        for i in range(len(self.hist)):
            k += self.hist[i][-1]
            normalizedHist.append(copy(self.hist[i]))
            normalizedHist[-1][-1] = float(k)
        for j in normalizedHist:
            #j[-1] = st.norm.ppf(float(j[-1])/float(self.total))
            j[-1] = ppf(float(j[-1])/float(self.total))
            
        return normalizedHist
    
    def getNormalizedValues(self, valores):
        normalizedHist = self.normalizedHistograma()
        
        #print "hist", self.hist
        #print "norm hist", normalizedHist
        
        if type(valores) != type([]):
            valores = [valores]
        
        results = []
        for i in valores:
            for j in normalizedHist:
                if i >= j[0]:
                    if i < j[1]:
                        
                        #if j[2] < -2.5: final = -3.09
                        #elif j[2] > 2.5: final = 3.09
                        final = j[2]
                         
                        results.append(final)
                        break
        #print results, normalizedHist
        return results
        