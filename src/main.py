import descriptorHOG
import cv2 as cv
import auxFunc as af
import pdb
import random
import numpy as np

'''
print("Obteniendo los datos de entrenamiento")
# Obtenemos los datos de entrenamiento
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
# Entrenamos la SVM
svm = descriptorHOG.trainSVM(td)
svm.save("svm.txt")
del td
'''
print("Cargando la SVM de fichero")
svm = cv.ml.SVM_load("svm.txt")

print("Cargandos el test")


imgs, _ = af.getImagesAndTags()

descr, tags = descriptorHOG.createTestData()
predicciones = svm.predict(descr)[1]
chunkedPred = descriptorHOG.chunkPredictions(imgs, predicciones)

del descr
del svm

print("\n\n##################################################")
print("Predicción: ")
print(predicciones)
print("##################################################\n\n")



npos=0
nneg=0
total_neg = 0
total_pos = 0

for t in tags:
    if t==1:
        total_pos+=1
    elif t==2:
        total_neg+=1

img_pos_correctas = []
img_pos_incorrectas = []
# Calculamos el número de aciertos
for i in range(len(predicciones)):
    if tags[i]==predicciones[i]:
        if predicciones[i]==1:
            npos+=1
            img_pos_correctas.append(np.uint8(imgs[i]))
        elif predicciones[i]==2:
            nneg+=1
    else:
        if predicciones[i]==2:
            img_pos_incorrectas.append(np.uint8(imgs[i]))
print("\n\n##################################################")
print("Positivos: " + str(npos) + "/" + str(total_pos) + "===>" + str(100*npos/total_pos) + "%")
print("Negativos: " + str(nneg) + "/" + str(total_neg) + "===>" + str(100*nneg/total_neg) + "%")
print("Porcentaje de acierto total: " + str(100*(npos+nneg)/(total_pos+total_neg)))
print("##################################################\n\n")

print("Pintando las imagenes positivas acertadas: ")
if len(img_pos_correctas)>50:
    af.pintaMI(random.sample(img_pos_correctas,50))
else:
    af.pintaMI(img_pos_correctas)

print("Pintando las imagenes positivas falladas: ")
if len(img_pos_incorrectas)>50:
    af.pintaMI(random.sample(img_pos_incorrectas,50))
else:
    af.pintaMI(img_pos_incorrectas)
