import csv

if __name__ == "__main__":
	file_name = "../data/output1.txt"
	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0
	
	with open(file_name, 'rb') as f:
		data = csv.reader(f)
		for v in data:
			if (v[1] == "True"):
				if (float(v[2]) >= 0.5):
					true_positives += 1
				else:
					false_positives += 1
			else:
				if (float(v[2]) >= 0.5):
					false_negatives += 1
				else:
					true_negatives += 1
	
	print("True positives: {0} | False positives: {1} \nTrue negatives: {2} | False negatives: {3}\nInstances: {4}".format(true_positives, false_positives, true_negatives, false_negatives, true_positives+false_positives+true_negatives+false_negatives))	
