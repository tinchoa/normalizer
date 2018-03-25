import csv
from hoeffdingtree import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


def open_dataset(filename, class_index, probe_instances=100):
		""" Open and initialize a dataset in CSV format.
		The CSV file needs to have a header row, from where the attribute names will be read, and a set
		of instances containing at least one example of each value of all nominal attributes.

		Args:
				filename (str): The name of the dataset file (including filepath).
				class_index (int): The index of the attribute to be set as class.
				probe_instances (int): The number of instances to be used to initialize the nominal 
						attributes. (default 100)

		Returns:
				Dataset: A dataset initialized with the attributes and instances of the given CSV file.
		"""
		if not filename.endswith('.csv'):
				raise TypeError(
						'Unable to open \'{0}\'. Only datasets in CSV format are supported.'
						.format(filename))
		with open(filename) as f:
				fr = csv.reader(f)
				headers = next(fr)

				att_values = [[] for i in range(len(headers))]
				instances = []
				try:
						for i in range(probe_instances):
								inst = next(fr)
								instances.append(inst)
								for j in range(len(headers)):
										try:
												inst[j] = float(inst[j])
												att_values[j] = None
										except ValueError:
												inst[j] = str(inst[j])
										if isinstance(inst[j], str):
												if att_values[j] is not None:
														if inst[j] not in att_values[j]:
																att_values[j].append(inst[j])
												else:
														raise ValueError(
																'Attribute {0} has both Numeric and Nominal values.'
																.format(headers[j]))
				# Tried to probe more instances than there are in the dataset file
				except StopIteration:
						pass

		attributes = []
		for i in range(len(headers)):
				if att_values[i] is None:
						attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
				else:
						attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

		dataset = Dataset(attributes, class_index)
		for inst in instances:
				for i in range(len(headers)):
						if attributes[i].type() == 'Nominal':
								inst[i] = int(attributes[i].index_of_value(str(inst[i])))
				dataset.add(Instance(att_values=inst))
		
		return dataset

def main():
		filename = 'last.csv'
		a=open(filename,'r')
		b=a.readlines()
		dados,label=dataPreparing(b)
		dataset = open_dataset(filename, 39, probe_instances=10000)
		vfdt = HoeffdingTree()
		
		# Set some of the algorithm parameters
		vfdt.set_grace_period(50)
		vfdt.set_hoeffding_tie_threshold(0.05)
		vfdt.set_split_confidence(0.0001)
		# Split criterion, for now, can only be set on hoeffdingtree.py file.
		# This is only relevant when Information Gain is chosen as the split criterion
		vfdt.set_minimum_fraction_of_weight_info_gain(0.01)

		vfdt.build_classifier(dataset)
		predict={}
		pre={}
		for i in range(len(dados)):
			new_divtest_instance = Instance(att_values=list(dados[i]))
			new_divtest_instance.set_dataset(dataset)
			predict[i]=vfdt.distribution_for_instance(new_divtest_instance)
			if predict[i][0] > 0.8:
				pre[i]='normal'
			else:
				pre[i]='threat'  
		a=pre.values()
		ra=np.asarray(label)
		print(accuracy_score(ra,a))
		print(precision_score(ra,a,labels=['normal','threat'], average='macro'))


if __name__ == '__main__':
		main()
