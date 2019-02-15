from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import math



def parseFileInput():

	words = []
	data = [] 
	
	#Reading File and removing the repetitions
	trainFile = open('./hw4data/traindata.txt', 'r')
	count = 0
	for line in trainFile:
		sentence = line.split()
		count += 1
		for loop in range(len(sentence)):
			if sentence[loop] not in words:
				words.append(sentence[loop])
		data.append(sentence)

	#Reading Stop list File
	stopList = []
	sFile = open('./hw4data/stoplist.txt', 'r')
	for stop in sFile:
		stopList.append(stop.split('\n')[0])

	#creating vocabulary
	vocabulary = []
	for train in range(len(words)):
		if words[train] not in stopList:
			vocabulary.append(words[train])

	#Sorting the vocabulary
	vocabulary = sorted(vocabulary)

	#Creation of features
	inputData =  [[0 for i in range(len(vocabulary))] for j in range(count)]
	trainLabels = []
	trainLabelFile = open('./hw4data/trainlabels.txt','r')
	for line in trainLabelFile:
		sentence = line.split('\n')
		trainLabels.append((int)(sentence[0]))

	for length in range(len(data)):
		for loop in range(len(data[length])):
			if data[length][loop] in vocabulary:
				index = vocabulary.index(data[length][loop])
				inputData[length][index] = 1

	sFile.close()
	trainLabelFile.close()
	trainFile.close()

	return inputData,trainLabels,vocabulary

def parseFileOutput(vocabulary):

	data = [] 
	testLabels = []
	
	#Reading Test File and test labels
	File = open('./hw4data/testdata.txt', 'r')
	count = 0
	for line in File:
		count += 1
		data.append(line.split())

	labelFile = open('./hw4data/testlabels.txt','r')
	for line in labelFile:
		testLabels.append((int)(line.split('\n')[0]))

	#Creation of feature space from vocabulary
	outputData =  [[0 for i in range(len(vocabulary))] for j in range(count)]

	for length in range(len(data)):
		for loop in range(len(data[length])):
			if data[length][loop] in vocabulary:
				outputData[length][vocabulary.index(data[length][loop])] = 1


	labelFile.close()
	File.close()

	return outputData,testLabels

def BayesClassifier(inputData,trainingClassLabel):

	global parameterCount
	parameterCount =len(inputData[0])
	probablityData =[[0 for i in range(parameterCount+1)] for j in range(2)]  # [0,i] is class == false, [1,i] is true.

	# constructor trains the classifier with probabilities of each parameter, given the class varaible
	featurecount = len(inputData)
	
	classTrueFeatureCount =0

	#print("featurecount",featurecount)
	
	# count where class variable is 1.
	for loop in range(featurecount):
		if trainingClassLabel[loop] == 1:
			classTrueFeatureCount += 1

	classFalseFeatureCount = featurecount- classTrueFeatureCount
	#print("featurecount",featurecount,"classTrueFeatureCount",classTrueFeatureCount,"classFalseFeatureCount",classFalseFeatureCount)

	probablityData[0][parameterCount] =(featurecount - classTrueFeatureCount +1 )/ (float)(featurecount +2);
	probablityData[1][parameterCount] = (classTrueFeatureCount + 1) / (float)(featurecount +2);

	
	#print("probablityDataOne",probablityData[1][parameterCount],"probablityDataZero",probablityData[0][parameterCount])
	#print(probablityData[1][parameterCount]+probablityData[0][parameterCount])

	for loop in range(parameterCount): 

		truecount = 0.0
		falsecount = 0.0
		#count where parameter == true and class == true or parameter == true and class == false
		#and divide by total with class == true or with class == false, respectively
		for example in range(featurecount):
			if inputData[example][loop] ==1 and trainingClassLabel[example] == 1:
				truecount += 1
			if inputData[example][loop] ==1 and trainingClassLabel[example] == 0:
				falsecount += 1

		#print(loop," ", truecount," ",falsecount)

		#Laplace Smoothing
		probablityData[0][loop] = (falsecount + 1)/(classFalseFeatureCount + classTrueFeatureCount + 2)
		probablityData[1][loop] = (truecount + 1)/(classFalseFeatureCount + classTrueFeatureCount + 2)

		#print(probablityData[0][loop], " ",probablityData[1][loop])

	global trained
	trained = 1

	#print (probablityData)

	return probablityData

def predictFeatureSet(inputData,trainingClassLabel,probablityData):

	predictedLables = []
	global trained
	if (trained == 0):
		print("The classifier must be trained before attempting to classify a feature set.");

	classIndex = len(inputData[0])

	# then fill in class varaible in new set of test features
	for loop in range(len(inputData)):

		fortune = math.log(probablityData[1][classIndex]);
		notFortune = math.log(probablityData[0][classIndex]);

		#print(fortune,notFortune)

		for feature in range(parameterCount):
			# print(loop," ",feature)
			if inputData[loop][feature] == 1:
				fortune += math.log(probablityData[1][feature])
			else:
				#rint(probablityData[1][feature])
				fortune += math.log(1 - probablityData[1][feature])

			if inputData[loop][feature] == 1:
				notFortune += math.log(probablityData[0][feature])
			else:
				#rint(probablityData[1][feature])
				notFortune += math.log(1 - probablityData[0][feature])

		#print("fortune",fortune,"notFortune",notFortune)

		if fortune > notFortune:
			predictedLables.append(1)
		else:
			predictedLables.append(0)

	#print(predictedLables,len(predictedLables))
	return predictedLables

def main():

	inputData,trainLabels,vocabulary = parseFileInput()
	outputData,testLabels = parseFileOutput(vocabulary)
	
	probablityData = BayesClassifier(inputData,trainLabels)

	print("\nTraining Data: ")
	#prediction on Training Set
	predictedLables = predictFeatureSet(inputData,trainLabels,probablityData)
	mattrain = confusion_matrix(trainLabels,predictedLables)
	print(mattrain)
	print("Accuracy: ",(mattrain[0,0]+mattrain[1,1])/(mattrain[0,0]+mattrain[1,1]+mattrain[1,0]+mattrain[0,1])*100)

	print("\nTesting Data: ")
	#prediction on Testing Set
	predictedLables = predictFeatureSet(outputData,testLabels,probablityData)
	mattest = confusion_matrix(testLabels,predictedLables)
	print(mattest)
	print("Accuracy: ",(mattest[0,0]+mattest[1,1])/(mattest[0,0]+mattest[1,1]+mattest[1,0]+mattest[0,1])*100)

	'''OUTPUT
	322 322
	[[161   9]
	 [  5 147]]
	Accuracy:  95.652173913
	101 101
	[[27 10]
	 [ 7 57]]
	Accuracy:  83.1683168317'''
	

	#Classification
	# model = MultinomialNB()
	# model.fit(inputData, trainLabels)

	# #Prediction on Training Data
	# predicted = model.predict(inputData)

	# mattrain = confusion_matrix(trainLabels, predicted)
	# print(mattrain)
	# print((mattrain[0,0]+mattrain[1,1])/(mattrain[0,0]+mattrain[1,1]+mattrain[1,0]+mattrain[0,1])*100) #97.5155279503

	# #Prediction on Test data
	# predicted = model.predict(outputData)

	# mattest = confusion_matrix(testLabels, predicted)
	# print(mattest)
	# print((mattest[0,0]+mattest[1,1])/(mattest[0,0]+mattest[1,1]+mattest[1,0]+mattest[0,1])*100) #65.3465346535

	'''OUTPUT

	GaussianNB
	[[162   8]
	 [  0 152]]
	97.5155279503
	[[11 26]
	 [ 9 55]]
	65.3465346535

	MultinomialNB
	[[162   8]
	 [  5 147]]
	95.9627329193
	[[27 10]
	 [ 7 57]]
	83.1683168317
	'''

trained = 0
parameterCount =0
main()


