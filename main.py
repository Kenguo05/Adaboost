from math import inf
import numpy as np
import pandas as pd
from scipy.special import expit

class Adaboost:
    def __init__(self, base=0):
        self.base = base
        self.baseList = []
        self.mean = []
        self.std = []

    def _preProcess(self, feature: np.ndarray):
        n = feature.shape[1]
        for i in range(n):
            m = feature[:, i].mean()
            s = feature[:, i].std()
            self.mean.append(m)
            self.std.append(s)
            feature[:, i] = (feature[:, i] - m) / s
        #for i in range(n):
        #    s = feature[:, i].sum()
        #    feature[:, i] /= s
        return feature
    
    def _loadData(self, dataPath='data.csv', labelPath='targets.csv'):
        feature = pd.read_csv(dataPath, header=None).values
        label = pd.read_csv(labelPath, header=None).values
        feature = self._preProcess(feature=feature)
        #print(np.shape(feature))
        return feature, label

    def _kFoldSplit(self, feature, label, k=10):
        setSize = int(np.floor(len(feature)/k))
        wholeIndex = [i for i in range(len(feature))]
        testIndex = np.random.choice(wholeIndex, setSize, replace=False)
        testFeature = feature[testIndex, :]
        testLabel = label[testIndex, :]
        trainFeature = np.delete(feature, testIndex, axis=0)
        trainLabel = np.delete(label, testIndex, axis=0)
        #print(np.shape(trainFeature))
        for i in range(len(testIndex)):
            testIndex[i] += 1
        return trainFeature, trainLabel, testFeature, testLabel, testIndex

    def _stumpClassifier(self, feature, col, threshold, flag):
        #print(np.shape(trainFeature))
        ansArray = np.ones((np.shape(feature)[0], 1))
        if flag == 0:
            ansArray[feature[:, col] <= threshold] = 0
        else:
            ansArray[feature[:, col] > threshold] = 0
        #print(np.shape(ansArray))
        return ansArray

    def _getStump(self, trainFeature, trainLabel: np.ndarray, D:np.ndarray):
        trainLabel = np.mat(trainLabel)
        trainFeature = np.mat(trainFeature)
        #print(np.shape(trainFeature))
        m, n = np.shape(trainFeature)
        stepNum = 10
        stump = {}
        stumpEst = np.zeros((m, 1))
        errorRate = inf
        for i in range(n):
            minVal = trainFeature[:, i].min()
            maxVal = trainFeature[:, i].max()
            step = (maxVal - minVal)/stepNum
            for j in range(stepNum+1):
                for f in [0, 1]:
                    threshold = minVal + float(j) * step
                    predictEst = self._stumpClassifier(feature=trainFeature, col=i, threshold=threshold, flag=f)
                    errArr = np.ones((m, 1))
                    #print(np.shape(trainLabel))
                    errArr[predictEst == trainLabel] = 0
                    weightedErr = D.T.dot(errArr)
                    if weightedErr <= errorRate:
                        errorRate = weightedErr
                        stumpEst = predictEst.copy()
                        stump['dimen'] = i
                        stump['threshold'] = threshold
                        stump['flag'] = f
        return stump, errorRate, stumpEst

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _func(self, X: np.ndarray, w: np.ndarray, b):
        z = X.dot(w) + b
        #print(z.shape)
        return self._sigmoid(z)

    #def _grad(self, X, y, w, b, D):
        #yz = np.multiply(y, self._func(X, w, b))
        #print(yz)
        #z = expit(yz)
        #print(z)
        #z0 = np.multiply((z - 1), y)
        #print(y*2-1)
        #z0 = np.multiply(z0, D)
        #print(z0)
        #gradW = X.T.dot(z0) + alpha * w
        #gradB = z0.sum()
        #return gradW, gradB

    def _getLR(self, trainFeature: np.ndarray, trainLabel: np.ndarray, D: np.ndarray):
        #trainFeature = self.kernel(trainFeature)
        trainFeature = np.mat(trainFeature)
        trainLabel = np.mat(trainLabel)
        w = np.random.normal(loc=0.0, scale=1.0, size=trainFeature.shape[1])
        b = np.random.normal(loc=0.0, scale=1.0)
        #print(np.shape(b))
        w = np.mat(w).reshape(-1, 1)
        #print(np.shape(w))
        m = np.shape(trainFeature)[0]
        learningRate = 1.0
        #print(Dw)
        LR = {}
        for i in range(50):
            yHat = self._func(trainFeature, w, b)
            yHat = np.mat(yHat).reshape(-1, 1)
            w -= learningRate * trainFeature.T.dot(np.multiply(D, (yHat - (trainLabel))))
            b -= learningRate * D.T.dot(yHat - (trainLabel))
        LR['w'] = w
        #print(np.shape(LR['w']))
        LR['b'] = b
        #print(np.shape(LR['b']))
        #print(w)
        #print(b)
        est = self._func(trainFeature, LR['w'], LR['b'])
        #print(z)
        #print(est)
        for j in range(len(est)):
            if est[j] >= 0.5:
                est[j] = 1
            else:
                est[j] = 0
        #print(est)
        errArr = np.ones((m, 1))
        errArr[est == trainLabel] = 0
        errorRate = D.T.dot(errArr)
        #print(errorRate)
        #print(est)
        return LR, errorRate, est

    def _verify(self, verifyFeature: np.ndarray, verifyLabel: np.ndarray, verifyIndex: list, tempModel: list, baseNum: int, foldNum: int):
        m = np.shape(verifyFeature)[0]
        testRe = np.zeros((m, 1))
        midVal = 0
        for i in range(len(tempModel)):
            if self.base == 1:
                classEst = self._stumpClassifier(verifyFeature, col=tempModel[i]['dimen'], threshold=tempModel[i]['threshold'], flag=tempModel[i]['flag'])
            #print(classEst)
                testRe += tempModel[i]['alpha'] * classEst
            #print(testRe)
            else:
                classEst = self._func(verifyFeature, tempModel[i]['w'], tempModel[i]['b'])
                testRe += tempModel[i]['alpha'] * classEst
                midVal += tempModel[i]['alpha'] * 0.5
        #print(testRe)
        if self.base == 1:
            midVal = (testRe.max() + testRe.min()) / 2
        #else:
        #    midVal = testRe.max() / 2
        #print(testRe)
        for j in range(len(testRe)):
            if testRe[j] < midVal:
                testRe[j] = 0
            else:
                testRe[j] = 1
        #print(testRe)
        testReDF = pd.DataFrame(testRe)
        testReDF.index = verifyIndex
        testReDF.to_csv('experiments/base%d_fold%d.csv' % (baseNum, foldNum), header=None)
        errLabel = np.ones((m, 1))
        errLabel[testRe == verifyLabel] = 0
        errRate = errLabel.sum() / m
        return errRate

    def fit(self, dataPath, labelPath):
        print("Waiting...")
        feature, label = self._loadData(dataPath=dataPath, labelPath=labelPath)
        minErr = inf
        for num in [1, 5, 10, 100]:
            for i in range(10):
                trainFeature, trainLabel, testFeature, testLabel, testIndex = self._kFoldSplit(feature=feature, label=label)
                weakList = []
                m = np.shape(trainFeature)[0]
                D = np.ones((m, 1))/m
                Dsum = m
                for j in range(num):
                    if self.base == 1:
                        model, errRate, est = self._getStump(trainFeature=trainFeature, trainLabel=trainLabel, D=D)
                    else:
                        model, errRate, est = self._getLR(trainFeature=trainFeature, trainLabel=trainLabel, D=D)
                    #print(est)
                    #print(errRate)
                    alpha = float(0.5 * np.log((1.0 - errRate)/errRate))
                    #print(alpha)
                    model['alpha'] = alpha
                    weakList.append(model)
                    for k in range(m):
                        if est[k] == trainLabel[k]:
                            D[k] *= np.exp((-1) * alpha)
                        else:
                            D[k] *= np.exp(alpha)
                    Dsum = D.sum()
                    D /= Dsum
                curErr = self._verify(verifyFeature=testFeature, verifyLabel=testLabel, verifyIndex=testIndex, tempModel=weakList, baseNum=num, foldNum=i+1)
                if curErr < minErr:
                    minErr = curErr
                    self.baseList = weakList
            print('Over on base%d' % (num))

    def predict(self, inputFile):
        inputX = pd.read_csv(inputFile, header=None).values
        m, n = np.shape(inputX)
        for i in range(n):
            inputX[:, i] = (inputX[:, i] - self.mean[i]) / self.std[i]
        predictResult = np.zeros((m, 1))
        midVal = 0
        for i in range(len(self.baseList)):
            if self.base == 1:
                classEst = self._stumpClassifier(inputX, self.baseList[i]['dimen'], self.baseList[i]['threshold'], self.baseList[i]['flag'])
                predictResult += self.baseList[i]['alpha'] * classEst
            else:
                classEst = self._func(inputX, self.baseList[i]['w'], self.baseList[i]['b'])
                predictResult += self.baseList[i]['alpha'] * classEst
                midVal += self.baseList[i]['alpha'] * 0.5
        if self.base == 1:
            midVal = (predictResult.min() + predictResult.max()) / 2
        for j in range(len(predictResult)):
            if predictResult[j] < midVal:
                predictResult[j] = 0
            else:
                predictResult[j] = 1
        return predictResult