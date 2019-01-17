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
svm = cv.ml.SVM_load("svm_sin_hardpositives.txt")

print("Cargando el test")

pred_pos, pred_neg, positivos_correctos, positivos_totales = af.getPredictions(svm)

del svm

print("\n\n##################################################")
print("Predicciones para las imagenes positivas: ")
print(pred_pos)
print("Predicciones para las imagenes negativas: ")
print(pred_neg)
print("##################################################\n\n")

pos_score = 0
neg_score = 0

for ppred in pred_pos:
    pos_score+=ppred

for npred in pred_neg:
    neg_score+=npred

print("\n\n##################################################")
print("Positivos: " + str(pos_score) + "/" + str(len(pred_pos)) + "===>" + str(100*pos_score/len(pred_pos)) + "%")
print("Negativos: " + str(neg_score) + "/" + str(len(pred_neg)) + "===>" + str(100*neg_score/len(pred_neg)) + "%")
print("Porcentaje de acierto total con la media por imagen: " + str(100*(pos_score+neg_score)/(len(pred_pos)+len(pred_neg))) + "%")
print("Peatones totales detectados: " + str(positivos_correctos) + "/" + str(positivos_totales) + "===>" + str(100*positivos_correctos/positivos_totales) + "%")
print("##################################################\n\n")
