import descriptorHOG
import cv2 as cv
import auxFunc as af

print("Obteniendo los datos de entrenamiento")
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
svm = descriptorHOG.trainSVM(td)
img_pos, img_neg = af.loadTestImgs()
img_pos_res = []
img_neg_res = []
for img in img_pos:
    img_pos_res.append(cv.resize(img,(64,128)))
for img in img_neg:
    img_neg_res.append(cv.resize(img,(64,128)))
print("Obteniendo los descriptores de las imagenes pregunta")
desc = descriptorHOG.obtainDescriptors(img_pos_res+img_neg_res)
predicciones = svm.predict(desc)

print("\n\n##################################################")
print("Predicción: ")
print(predicciones)
print("##################################################\n\n")


npos=0
nneg=0
for i in range(len(predicciones[1])):
    if i<len(img_pos):
        if predicciones[1][i]==1:
            npos+=1
    else:
        if predicciones[1][i]==-1:
            nneg+=1
print("\n\n##################################################")
print("Positivos: " + str(npos) + "/" + str(len(img_pos)))
print("Negativos: " + str(nneg) + "/" + str(len(img_neg)))
print("##################################################\n\n")
