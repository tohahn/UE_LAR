import numpy as np
import matplotlib.pyplot as plt

from KMeans import KMeans

class Tests:
	
	def test(self, k, data, random=False):
		kmeans = KMeans()
		book = kmeans.iterate(k, data, random)
		print('---Start---')
		for i in range(len(book)):
			b = book[i]
			print(b[0])
			listX = []
			listY = []
			repVecX = [b[0][0]]
			repVecY = [b[0][1]]
			for vec in b[1]:
				listX.append(vec[0])
				listY.append(vec[1])
			plt.plot(listDX, listDY, 'ro', listX, listY, 'g^', repVecX, repVecY, 'bs')
			plt.axis([-5, 30, -5, 30])
			plt.show()
			plt.clf()
		print('---End---')

if __name__ == '__main__':
	t = Tests()
	
	data = np.loadtxt("./cluster_dataset2d.txt")
	
	listDX = []
	listDY = []
	
	for d in data:
		listDX.append(d[0])
		listDY.append(d[1])
	
	t.test(3, data)
	t.test(6, data)
	t.test(12, data)
	t.test(3, data, True)
	t.test(6, data, True)
	t.test(12, data, True)