import descriptorHOG
import cv2 as cv
import auxFunc as af
import pdb
import random
import numpy as np

print("Obteniendo los datos de entrenamiento")
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
svm = descriptorHOG.trainSVM(td)
del td
print("Cargandos las imagenes de test")
img_pos, img_neg = af.loadTestImgs()
img_pos = random.sample(img_pos,100)
img_neg = random.sample(img_neg,100)
img_pos_res = 0
img_neg_res = 0
predicciones = []
print("Obteniendo las ventanas y descriptores de las imágenes pregunta")
imagenes_positivas = []
for img in img_pos:
    windows = random.sample(af.getAllWindows(img),20)
    desc = descriptorHOG.obtainDescriptors(windows,True)
    img_pos_res+=len(windows)
    contador=0
    for pred in svm.predict(desc)[1]:
        predicciones.append(pred[0])
        if pred[0]==1:
            imagenes_positivas.append(np.uint8(windows[contador]))
        contador+=1
for img in img_neg:
    windows = random.sample(af.getAllWindows(img),20)
    desc = descriptorHOG.obtainDescriptors(windows,True)
    img_neg_res+=len(windows)
    for pred in svm.predict(desc)[1]:
        predicciones.append(pred[0])

del img_pos
del img_neg

print("\n\n##################################################")
print("Predicción: ")
print(predicciones)
print("##################################################\n\n")


npos=0
nneg=0
for i in range(len(predicciones)):
    if i<img_pos_res:
        if predicciones[i]==1:
            npos+=1
    else:
        if predicciones[i]==2:
            nneg+=1
print("\n\n##################################################")
print("Positivos: " + str(npos) + "/" + str(img_pos_res))
print("Negativos: " + str(nneg) + "/" + str(img_neg_res))
print("##################################################\n\n")

af.pintaMI(imagenes_positivas)
