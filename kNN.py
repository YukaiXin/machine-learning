#kNN.py

from numpy import  *
import operator

def createDataSet():

    group = array([[1.0, 1.1, 1.2], [1.2, 1.0, 1.1], [1.1, 1.2, 1.0], [0.1, 0.5, 0.7], [0.2, 0,1, 0.3], [2.1, 2.0, 2.1]])
    labels  = ["AAA", "AAA", "BBB", "BBB", "CCC"]

    return group,labels


def classfity(sampleX, dataSet, labels, k):

    dataSetSize = len(dataSet)

    diffMat = tile(sampleX, (dataSetSize, 1))  - dataSet

    sqDiffMat = diffMat ** 2

    sqDistances = sqDiffMat.sum(axis =1)

    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsorts()

    classCount = {}

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]




