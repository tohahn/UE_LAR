import numpy as np

class KMeans:
	codeBook=None
	
	def iterate(self, k, sample, random=False):
		self.codeBook = []
	
		sample = self.createStartReproductionVecs(k, sample, random)
		
		for i in range(len(sample)):
			x = sample[i]
			
			rIndex = self.getReproductionMinDistIndex(x)
			r = self.codeBook[rIndex][1]
			
			r.append(x)
			
			cen = self.calcNewCentrum(rIndex)
			self.codeBook[rIndex][0] = cen
			
		return self.codeBook
		
	def createStartReproductionVecs(self, k, sample, random):
		if random:
			for i in range(k):
				index = np.random.randint(0, high=len(sample))
				self.codeBook.append([sample[index], [sample[index]]])
				sample1 = sample[0:index-1]
				sample2 = sample[index+1:len(sample)]
				np.concatenate((sample1, sample2), axis=0)
		else:
			for i in range(k):
				self.codeBook.append([sample[0], [sample[0]]])
				sample = sample[1:len(sample)]
		
		return sample
		
	def calcNewCentrum(self, index):
		means = self.calcMeanVector(self.codeBook[index][1])
		cent = self.getVectorMinDistInCell(means, self.codeBook[index][1])
		return cent
		
	def getReproductionMinDistIndex(self, vec):
		index = 0
		distMin = self.calcEuclideanDistance(vec, self.codeBook[0][0])
		for i in range(1, len(self.codeBook)):
			distComp = self.calcEuclideanDistance(vec, self.codeBook[i][0])
			if distMin > distComp:
				index = i
				distMin = distComp
		
		return index
	
	def getVectorMinDistInCell(self, vec, cell):
		vecRet = cell[0]
		distMin = self.calcEuclideanDistance(vec, cell[0])
		for i in range(1, len(cell)):
			distComp = self.calcEuclideanDistance(vec, cell[i])
			if distMin > distComp:
				vecRet = cell[i]
				distMin = distComp
			
		return vecRet
	
	def calcMeanVector(self, vectors):
		vecLength = len(vectors[0])
		means = [0 for x in range(vecLength)]
		for i in range(len(vectors)):
			for j in range(vecLength):
				means[j] += vectors[i][j]
		
		for i in range(vecLength):
			means[i] = means[i] / len(vectors)
		
		return means
	
	def calcEuclideanDistance(self, vec1, vec2):
		return np.linalg.norm(vec1 - vec2)