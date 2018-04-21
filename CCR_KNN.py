# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:36:43 2017

@author: Administrator
"""
from numpy import * #导入numpy的函数库
import numpy as np

import scipy.io as scio
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
from matplotlib import colors
from sklearn.lda import LDA
import h5py as h5
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn import neighbors as ng
from sklearn.svm import SVC 

def getTrainAndTestSet(path,trainNum, dataName, lion_type, huamn_type):
	'get trainset and testset'
	Data = scio.loadmat(path) #for GLCM first
	All = Data[dataName]
#	if (dataName == 'GLCMData') and (huamn_type == 'glcm_SIT' ) : #for GLCM second
#		Data = scio.loadmat(path)
#		All = Data[dataName]
#	else:
#		Data = h5.File(path)
#		All = Data[dataName]
#		All = np.array(All)
#		All = All.T

#	if huamn_type == "glcm_human_lamp":
#		Data = h5.File(path)
#		All = Data[dataName] #
#		All = np.array(All)
#		All = All.T
#		
#	else:
#		Data = scio.loadmat(path)
#		All = Data[dataName]
	#Data = scio.loadmat(path)
	#All = Data[dataName]
	#this part for lions: mat v.7.3
#	Data = h5.File(path)
#	All = Data[dataName] #
#	All = np.array(All)
#	All = All.T


	#lgbpLionsData = h5.File(path,'r')
	#print (lgbpLionsData)
	#All = Data[dataName] #
	#All = np.array(All)
	#print(type(All))
	#print(shape(All))
	#All = All.T
	#获取矩阵的LGBP特征
	Features = All[0:shape(All)[0],:] #384x1
	
	#获取矩阵的最后一列,-1为倒数第一 shape(lgbpLionsAll)[0]得到行数,shape(lgbpLionsAll)[1]得到列数
	#lgbpLionsName = lgbpLionsAll[0:shape(lgbpLionsAll)[0],-1:shape(lgbpLionsAll)[1]]
	#lions train 
	trainSet = Features[0:trainNum,:] #200x1

	#lion test
	testSet = Features[trainNum:shape(All)[0],:] #183x1
	#lionsTestSetName = lgbpLionsName[trainNum+1:shape(lgbpLionsAll)[0],:]
	return trainSet,testSet

def getTrainAndTestSetRace(path,trainNum, dataName):
	'get trainset and testset'
	Data = h5.File(path)
	All = Data[dataName] #
	All = np.array(All)
	All = All.T

	Features = All[0:shape(All)[0],:] #384x1
	
	#获取矩阵的最后一列,-1为倒数第一 shape(lgbpLionsAll)[0]得到行数,shape(lgbpLionsAll)[1]得到列数
	#lgbpLionsName = lgbpLionsAll[0:shape(lgbpLionsAll)[0],-1:shape(lgbpLionsAll)[1]]
	#lions train 
	trainSet = Features[0:trainNum,:] #200x1

	#lion test
	testSet = Features[trainNum:shape(All)[0],:] #183x1
	#lionsTestSetName = lgbpLionsName[trainNum+1:shape(lgbpLionsAll)[0],:]
	return trainSet,testSet



def LDAClassificationForIris(trainNum, _type, *dataSet):
	'This function is for LDA classification'
	print('kkkkkkkkkkkkkkkkkkkkkkkk')
	#print(dataSet[0])
	
	
	
	trainSet = np.concatenate((dataSet[0],dataSet[2]),axis=0) #lion + all human

	trainLabelOne = np.zeros((shape(dataSet[0])[0],1)) #first class label for lions
	trainLabelTwo = np.ones((shape(dataSet[2])[0],1)) #second class label for all human
	trainLabel = np.concatenate((trainLabelOne,trainLabelTwo),axis=0)

	testLabelOne = np.zeros((shape(dataSet[1])[0],1))
	testLabelTwo = np.ones((shape(dataSet[3])[0],1))
	testLabel = np.concatenate((testLabelOne, testLabelTwo),axis=0)
	#print (shape(testLabel)) #417x1
	testSetOne = np.array(dataSet[1])
	testSetTwo = np.array(dataSet[3])
	testSet = np.concatenate((testSetOne,testSetTwo),axis=0) #testSet : 417x2360
#	print ('++++++++++++++++++++')
#	print (shape(trainSet))
#	print (shape(trainLabel))
	print ('------------------------------')
	#print (trainSet.shape)
	#print (trainLabel.shape)
	clf = ng.KNeighborsClassifier(algorithm='kd_tree')
	clf.fit(trainSet, trainLabel)
#	SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\
#		decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',\
#		max_iter=-1, probability=False, random_state=None, shrinking=True,\
#		tol=0.001, verbose=False)
	
	print ('=========================The classification results are :=======================')
	classificationResult = clf.predict(testSet)
	#存储分类结果: 已知类别  分类类别
	#print(testLabel.shape), print(classificationResult.shape)
	#classificationResult.shape=(testLabel.shape[0],1)
	#print(testLabel.shape), print(classificationResult.shape)
	origin_class = testLabel
	clf_class = classificationResult
	clf_class.shape = (testLabel.shape[0],1)
	all_class = np.concatenate((origin_class, clf_class), axis=1)
	print('---------------测试集的个数--------------')
	print(all_class.shape[0])
	#label_name = _type + '_SVM_ORIGIN_CLF_LABEL.csv'
	#np.savetxt(label_name, all_class, delimiter = ',')  
	print ('=========the classfiy results==============') #1x417; 
	print (classificationResult)

	#save the classificationResult: the first column is true label, the second is classification label
	#testLabel.T: 转置; testLabel为1x417,classificationResult为417x1,维数不同,需要化为相同
	
	trueLabelAndClassifyLabel = np.concatenate((testLabel,classificationResult),axis=0)
	trueLabelAndClassifyLabel = trueLabelAndClassifyLabel.T
	#print (trueLabelAndClassifyLabel.shape)
	count = 0
	wrong_num = 0
	for i in range(1,shape(classificationResult)[0]):
		if testLabel[i] == classificationResult[i]:
			count = count + 1
		else:
			wrong_num += 1
			print('=============the wrong num is ' + str(i))
	print('一共有' + str(wrong_num) + '幅图分类错误')
	accurcay = count/classificationResult.shape[0]
	print ('======================The accurcay of LDA:==========================================')
	print (accurcay)

	print ('======================The scores:===============================================')
	weight = [0.0001 for i in range(classificationResult.shape[0])]
	for x in range(1,classificationResult.shape[0]):
		weight[x-1] = random.uniform(0,1)
	print(clf.score(testSet, testLabel,weight))

	#print ('======================The Estimate probability=================================')
	#estimate_pro = clf.predict_proba(testSet) # for get ROC
	#print (estimate_pro)
	#print (estimate_pro.shape)

	#print ('======================Predicit confidence scores for samples:============================')
	#predicit_confidence = clf.decision_function(testSet)
	#print (predicit_confidence)
	#print (predicit_confidence.shape)
	#call ROC
	#yLabel = np.concatenate((trainLabel,testLabel),axis=0)
	#getROCCurve(testLabel, predicit_confidence)
	#交叉验证
	X = np.concatenate((trainSet,testSet),axis=0)
	Y = np.concatenate((trainLabel,testLabel),axis=0)
	YY = Y
	YY.shape = (YY.shape[0],)
	#kFold = cross_validation.KFold(len(X),6, shuffle=True)
	kFold = StratifiedKFold(YY, n_folds=6)
	
	acc_y = getROCCurve(clf,X, Y, kFold, _type)
	print ('======================The terminal acc_y of LDA:==========================================')
	print(acc_y)
	#return acc_y #for CCR
	return accurcay






def lgbpForIrisLDA(lion_path, human_path,trainNum, _type, lion_type, huamn_type):
	
	trainNum = 10000
	#../表示上一级目录
	lgbpLions = lion_path#'../../big_data_feature_extraction/LGBP/matrixLGBP/LGBPRotateLions/LGBPRotateLionsFeature.mat'
	lgbpHuman = human_path#'../../big_data_feature_extraction/LGBP/matrixLGBP/LGBPThousand/LGBPThousandFeature.mat'
	#lgbpHumanGlass = '../../feature_extraction/matrixLGBP/LGBPHumanGlass.mat'
	#label
	lionLabel = 0;
	humanLabel = 1;
	#humanGlassLabel = 1;
	#print(shape(lgbpLions))
	#for lions
	(lionsTrainSet,lionsTestSet) = getTrainAndTestSet(lgbpLions,trainNum,'LGBPData',lion_type, huamn_type)
	print('=====================================================================')
	print(shape(lionsTrainSet))
	print(type(lionsTrainSet))


	#for human 
	(humanTrainSet,humanTestSet) = getTrainAndTestSet(lgbpHuman,trainNum,'LGBPData',lion_type, huamn_type)
	#shape(lionsTrainSet)
	#for humanglass
	#(humanGlassTrainSet,humanGlassTestSet) = getTrainAndTestSet(lgbpHumanGlass,trainNum,'LGBPHumanGlass')
	#print (type(humanGlassTrainSet))
	#print (shape(humanGlassTrainSet))
	accurcay = LDAClassificationForIris(trainNum,_type, lionsTrainSet,lionsTestSet, \
		humanTrainSet, humanTestSet)
	return accurcay


	

def glcmForIrisLDA(lion_path, human_path,trainNum, _type, lion_type, huamn_type):
	
	#../表示上一级目录
	glcmLions = lion_path#'../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat'
	glcmHuman = human_path#'../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMThousand/GLCMThousandFeature.mat'
	#glcmHumanGlass = '../../feature_extraction/matrixGLCM/GLCMHumanGlass.mat'
	
	#for lions
	(glcmLionsTrainSet,glcmLionsTestSet) = getTrainAndTestSet(glcmLions,trainNum,'GLCMData', lion_type, huamn_type)
#	print('=======================kenanananana--------------------')
#	print(glcmLionsTrainSet.shape) 
#	print(glcmLionsTestSet.shape)
	#for human 
	(glcmHumanTrainSet,glcmHumanTestSet) = getTrainAndTestSet(glcmHuman,trainNum,'GLCMData', lion_type, '123')
#	print(glcmHumanTrainSet.shape)
#	print(glcmHumanTestSet.shape)
	#for humanglass
	#(glcmHumanGlassTrainSet,glcmHumanGlassTestSet) = getTrainAndTestSet(glcmHumanGlass,trainNum,'GLCMHumanGlass')
	#print (type(humanGlassTrainSet))
	#print (shape(humanGlassTrainSet))
	accurcay = LDAClassificationForIris(trainNum,_type, glcmLionsTrainSet,glcmLionsTestSet, \
		glcmHumanTrainSet, glcmHumanTestSet)
	return accurcay

def forRaceLDA(asian_path,  white_path,train_num, _type, file_type):
	
	(AsianTrainSet,AsianTestSet) = getTrainAndTestSetRace(asian_path,train_num, _type+'AsianTrain')
	(WhiteTrainSet,WhiteTestSet) = getTrainAndTestSetRace(white_path,train_num, _type+'WhiteTrain')

	acc_y = LDAClassificationForIris(train_num,file_type, AsianTrainSet,AsianTestSet, \
		WhiteTrainSet, WhiteTestSet)
	return acc_y



def getROCCurve(clf, X, Y, kFold, _type):
	print ('====================================get ROC ====================')
	#交叉验证
	mean_tpr = 0.0
	mean_fpr = np.linspace(0,1,100)
	#the accuracy
	acc_y = []
	for i, (trn,tst) in enumerate(kFold):
		#print (tst)
		proBas = clf.fit(X[trn], Y[trn]).predict_proba(X[tst])
		#通过roc_curve()函数求出fpr,tpr,以及阈值
		fpr,tpr,thresholds = roc_curve(Y[tst], proBas[:,1])
		mean_tpr += interp(mean_fpr,fpr,tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr,tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
		outVal = clf.score(X[tst], Y[tst])
		acc_y.append(outVal)
		#print (outVal)
	#	plt.plot(fpr, tpr, lw=1, label='ROC')
	#acc_y = np.mean(acc_y)
	print('========每一次的acc_y===========--------------------')
	print(acc_y)
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	#plt.plot(fpr, tpr, lw=1, color='#FF0000', label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	mean_tpr /= len(kFold)
	mean_tpr[-1] = 1.0 						#坐标最后一个点为（1,1）
	mean_auc = auc(mean_fpr, mean_tpr)		#计算平均AUC值
	#画平均ROC曲线
	#print mean_fpr,len(mean_fpr)
	#print mean_tpr: 存储fpr,tpr: EER: ROC与ROC空间对角线交点的横坐标
	print('------------fpr, tpr-----------------')
	mean_fpr1 = mean_fpr
	mean_fpr1.shape = (mean_fpr.shape[0],1)
	#print(mean_fpr1.shape)
	mean_tpr1 = mean_tpr
	mean_tpr1.shape = (mean_tpr.shape[0],1)
	#print(mean_tpr1.shape)
	fpr_tpr = np.concatenate((mean_fpr1,mean_tpr1),axis=1)
	#roc_data_name = _type + '_FPR_TPR.csv'
	#np.savetxt(roc_data_name, fpr_tpr, delimiter = ',')
	#print(mean_fpr.shape)
	#print(mean_tpr.shape)
	#for EER
	for i in range(mean_fpr.shape[0]):
		if mean_fpr[i] == mean_tpr[i]:
			eer = mean_fpr[i]
			break;
	print('--------------------eer------------' )
	print(eer)

	plt.plot(mean_fpr, mean_tpr,  '--',color='#0000FF',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=1)
	plt.xlim([-0.02, 1.02])
	plt.ylim([-0.02, 1.02])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	return np.mean(acc_y)



if __name__ == "__main__":

	loop_num=np.array([100,1000,2000,4000,6000,8000,10000])
	acc=[]
	#first: lions:thousand
	#LGBP
#	for i in range(len(loop_num)):
#		print ('++++++++++++++++LGBP IRIS 测试集个数为:' + str(loop_num[i])+ '+++++++++++++++++++++++')
#		#glcmLions = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat'
#		#glcmHuman = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMThousand/GLCMThousandFeature.mat'
#		lgbp_lions_path =  '../big_data_feature_extraction/LGBP/matrixLGBP/LGBPRotateLions/LGBPRotateLionsFeature.mat' #19200
#		lgbp_human_path = '../big_data_feature_extraction/LGBP/matrixLGBP/LGBPThousand/LGBPThousandFeature.mat' #20000
#		trainNum = loop_num[i]
#		accurcay = lgbpForIrisLDA(lgbp_lions_path, lgbp_human_path,train_num, 'LGBP_Lions_Thousand','lgbp_lions', 'lgbp_human')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('LGBP_Lions_Thousand_CCR_KNN.csv', acc, delimiter = ',') 
	#GLCM
#	for i in range(len(loop_num)):
#		print ('++++++++++++++++GLCM IRIS 测试集个数为:' + str(loop_num[i])+ '+++++++++++++++++++++++')
#		#glcmLions = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat'
#		#glcmHuman = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMThousand/GLCMThousandFeature.mat'
#		glcm_lions_path =  '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat' #19200
#		glcm_human_path = '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMThousand/GLCMThousandFeature.mat' #20000
#		trainNum = loop_num[i]
#		accurcay = glcmForIrisLDA(glcm_lions_path, glcm_human_path, trainNum,'GLCM_Lions_Thousand','glcm_lions', 'glcm_thousand')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('GLCM_Lions_Thousand_CCR_KNN.csv', acc, delimiter = ',')  
#


	#second: lamp:lions
	#LGBP
#	for i in range(len(loop_num)):
#		print ('++++++++++++++++LGBP IRIS 测试集个数为:' + str(loop_num[i])+ '+++++++++++++++++++++++')
#		#glcmLions = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat'
#		#glcmHuman = '../../big_data_feature_extraction/GLCM/matrixGLCM/GLCMThousand/GLCMThousandFeature.mat'
#		lgbp_lions_path =  '../../big_data_feature_extraction/LGBP/matrixLGBP/LGBPRotateLions/LGBPRotateLionsFeature.mat' #19200
#		lgbp_human_path = '../big_data_feature_extraction/LGBP/matrixLGBP/LGBPLamp/LGBPLampFeature.mat'
#		trainNum = loop_num[i]
#		accurcay=lgbpForIrisLDA(lgbp_lions_path, lgbp_human_path,train_num, 'LGBP_Lions_Lamp','lgbp_lions', 'lgbp_lamp')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('GLCM_Lions_Lamp_CCR_KNN.csv', acc, delimiter = ',') 
	#GLCM
#	for i in range(len(loop_num)):
#		print ('++++++++++++++++GLCM IRIS 测试集个数为:' + str(loop_num[i])+ '+++++++++++++++++++++++')
#		glcm_lions_path =  '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat' 
#		glcm_human_path = '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMLamp/GLCMLampFeature.mat'
#		trainNum = loop_num[i]
#		accurcay = glcmForIrisLDA(glcm_lions_path, glcm_human_path, trainNum,'GLCM_Lions_Lamp','glcm_lions', 'glcm_lamp')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('GLCM_Lions_Lamp_CCR_KNN.csv', acc, delimiter = ',')  

	#third
	#GLCM
#	for i in range(len(loop_num)):
#		print ('++++++++++++++++GLCM IRIS 测试集个数为:' + str(loop_num[i])+ '+++++++++++++++++++++++')
#		glcm_lions_path =  '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMRotateLions/GLCMRotateLionsFeature.mat' 
#		glcm_human_path = '../big_data_feature_extraction/GLCM/matrixGLCM/GLCMSIT/GLCMSITFeature.mat'
#		trainNum = loop_num[i]
#		accurcay = glcmForIrisLDA(glcm_lions_path, glcm_human_path, trainNum,'GLCM_Lions_SIT','glcm_lions', 'glcm_SIT')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('GLCM_Lions_SIT_CCR_KNN.csv', acc, delimiter = ',')  
#
#	print('=====ccr=====') #训练集10000表示每类中选取10000个作为训练
#	print(acc)
#	plt.plot(loop_num*2,acc, 'r--*')
#	plt.xlabel('Number of iris images userd for training per each class')
#	plt.ylabel('Correct Classification Rate')

	#fourth:
	loop_num1=np.array([100,200,300,400,500,600,700])
	#LGBP
	for i in range(len(loop_num1)):
		print ('++++++++++++++++GLCM RACE 测试集个数为:' + str(loop_num1[i])+ '+++++++++++++++++++++++')
		lgbp_asian_path = '../RACE_classification/Race_Data/LGBPAsian.mat'
		lgbp_white_path = '../RACE_classification/Race_Data/LGBPWhite.mat'
		trainNum = loop_num1[i]
		accurcay = forRaceLDA(lgbp_asian_path,lgbp_white_path, trainNum,'LGBP','LGBP_Asian_White')
		print('==============accurcay:')
		print(accurcay)
		acc.append(accurcay)
	np.savetxt('LGBP_RACE_CCR_KNN.csv', acc, delimiter = ',')  
	
	#GLCM
#	for i in range(len(loop_num1)):
#		print ('++++++++++++++++GLCM RACE 测试集个数为:' + str(loop_num1[i])+ '+++++++++++++++++++++++')
#		glcm_asian_path = '../RACE_classification/Race_Data/GLCMAsian.mat'
#		glcm_white_path = '../RACE_classification/Race_Data/GLCMWhite.mat'
#		trainNum = loop_num1[i]
#		accurcay = forRaceLDA(glcm_asian_path,glcm_white_path, trainNum,'GLCM','GLCM_Asian_White')
#		print('==============accurcay:')
#		print(accurcay)
#		acc.append(accurcay)
#	np.savetxt('GLCM_RACE_CCR_KNN.csv', acc, delimiter = ',')  

	print('=====ccr=====') #训练集10000表示每类中选取10000个作为训练
	print(acc)
	plt.plot(loop_num1*2,acc, 'r--*')
	plt.xlabel('Number of iris images userd for training per each class')
	plt.ylabel('Correct Classification Rate')








