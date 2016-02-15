import math
import operator
from string import split
import json
import pdb

def majorityCnt(classlist):
	classcount = {}
	for v in classlist:
		if v not in classcount.keys():
			classcount[v] = 0
		classcount[v] += 1
	sortedClassCount = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def entropy(dataset):
	n = len(dataset)
	labels = {}
	for record in dataset:
		label = record[-1]
		if label not in labels.keys():
			labels[label] = 0
		labels[label] += 1
	ent = 0.0
	for key in labels.keys():
		prob = float(labels[key]) / n
		ent += - prob * math.log(prob, 2)
	return ent

def splitDataset(dataset, colnum, value):
	retDataSet = []
	for record in dataset:
		if record[colnum] == value:
			reducedRecord = record[:colnum]
			reducedRecord.extend(record[colnum + 1 :])
			retDataSet.append(reducedRecord)
	return retDataSet

def chooseBestFeatureToSplit(dataset):
	numberFeature = len(dataset[0]) - 1
	baseEntropy = entropy(dataset)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numberFeature):
		featureList = [x[i] for x in dataset]
		uniqueValues = set(featureList)
		newEntropy = 0.0
		for value in uniqueValues:
			subDataset = splitDataset(dataset, i, value)
			prob = len(subDataset)/float(len(dataset))
			newEntropy += prob * entropy(subDataset)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def buildTree(dataset, labels):
	classlist = [x[-1] for x in dataset]
	if classlist.count(classlist[0]) == len(classlist):
		return classlist[0]
	if len(dataset[0]) == 1:
		return majorityCnt(classlist)
	bestFeature = chooseBestFeatureToSplit(dataset)
	bestFeatureLabel = labels[bestFeature]
	tree = {bestFeatureLabel:{}}
	del(labels[bestFeature])
	featValues = [x[bestFeature] for x in dataset]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		nextdataset = splitDataset(dataset, bestFeature, value)
		tree[bestFeatureLabel][value] = buildTree(nextdataset, subLabels)
	return tree

def outputFunc(tree):
	data = tree
	f = open('tree.txt', 'w')
	f.write(json.dumps(data, sort_keys=True, indent=4))
	f.close()

def classify(tree, labels, test):
	firstStr = tree.keys()[0]
	secondDict = tree[firstStr]
	featIndex = labels.index(firstStr)
	for key in secondDict.keys():
		if test[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], labels, test)
			else:
				classLabel = secondDict[key]
	try:
		return classLabel
	except:
		return 1


fs = open("Training.data")
dataset = []
for line in fs:
	lineSplit = split(line[:-1], ",")
	dataset.append([value for value in lineSplit])
fs.close()

num_of_feature = len(dataset[0])
labels = ["att"+str(i) for i in range(num_of_feature - 1)]
labels2 = [x for x in labels]
tree = buildTree(dataset, labels)
outputFunc(tree)

fs = open("Testing.data")
testDataset = []
for line in fs:
	lineSplit = split(line[:-1], ",")
	testDataset.append([value for value in lineSplit])
fs.close()

nPos = 0
for r in testDataset:
	ret = classify(tree, labels2, r)
	if ret == r[-1]:
		nPos += 1
num_test = len(testDataset)
pass_rate = nPos / float(num_test)
f = open('tree.txt', 'a+')
f.write("\nThe pass rate is " + str(pass_rate))
f.close()