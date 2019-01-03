import descriptorHOG
import cv2 as cv
import auxFunc as af
import pdb

print("Obteniendo los datos de entrenamiento")
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
svm = descriptorHOG.trainSVM(td)
print("Cargandos las imagenes de test")
img_pos, img_neg = af.loadTestImgs()
img_pos_res = 0
img_neg_res = 0
predicciones = []
print("Obteniendo las ventanas y descriptores de las imágenes pregunta")
af.pintaMI(img_pos[:3])
for img in img_pos[:3]:
    windows = af.getAllWindows(img)[:5]
    desc = descriptorHOG.obtainDescriptors(windows)
    img_pos_res+=len(windows)
    af.pintaMI(windows)
    pdb.set_trace()
    for pred in svm.predict(desc)[1]:
        predicciones.append(pred[0])
for img in img_neg[:3]:
    windows = af.getAllWindows(img)[:5]
    desc = descriptorHOG.obtainDescriptors(windows)
    img_neg_res+=len(windows)
    for pred in svm.predict(desc)[1]:
        predicciones.append(pred[0])

print("\n\n##################################################")
print("Predicción: ")
print(predicciones)
print("##################################################\n\n")


npos=0
nneg=0
for i in range(len(predicciones)):
    if i<img_pos_res:
        if predicciones[i]==3:
            npos+=1
    else:
        if predicciones[i]==4:
            nneg+=1
print("\n\n##################################################")
print("Positivos: " + str(npos) + "/" + str(img_pos_res))
print("Negativos: " + str(nneg) + "/" + str(img_neg_res))
print("##################################################\n\n")
