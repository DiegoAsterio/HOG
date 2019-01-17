import descriptorHOG
import cv2 as cv
import auxFunc as af
import pdb
import random
import numpy as np

'''
print("Obteniendo los datos de entrenamiento")
# Obtenemos los datos de entrenamiento
td = descriptorHOG.obtainHardTrainData()
print("Entrenando la SVM")
# Entrenamos la SVM
svm = descriptorHOG.trainSVM(td)
svm.save("svm.txt")
exit()
del td
'''
print("Cargando la SVM de fichero")
svm = cv.ml.SVM_load("svm.txt")

print("Cargando el test")

pred_pos, pred_neg, positivos_correctos, positivos_totales = af.getPredictions(svm)

del svm

print("\n\n##################################################")
print("Predicciones para las imagenes positivas: ")
print(pred_pos)
print("Predicciones para las imagenes negativas: ")
print(pred_neg)
print("##################################################\n\n")

af.printStats(pred_pos, pred_neg, positivos_correctos, positivos_totales)
