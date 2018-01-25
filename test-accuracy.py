
normalized=[]
for i in proposal:
	normalized.append(i.normalizedHistograma())

valuesProposta={}                                     
for i in range(len(normalized)):                      
    valuesProposta[i]=[]                              
    tmp=[]                                            
    for k in range(len(normalized[i])):               
        tmp.append(normalized[i][k][2])               
    valuesProposta[i].append(np.mean(np.array(tmp)))  
    valuesProposta[i].append(np.std(np.array(tmp)))   


valuesOriginal={}                                     
for i in range(len(histogramOriginal)):                      
    valuesOriginal[i]=[]                              
    tmp=[]                                            
    for k in range(len(histogramOriginal[i])):               
        tmp.append(histogramOriginal[i][k][2])               
    valuesOriginal[i].append(np.mean(np.array(tmp)))  
    valuesOriginal[i].append(np.std(np.array(tmp)))   



