import descriptorHOG
import cv2 as cv

print("Obteniendo los datos de entrenamiento")
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
svm = descriptorHOG.trainSVM(td)
img = cv.resize(cv.imread("../../INRIAPerson/Test/pos/crop_000001.png",-1),(64,128))
img2 = cv.resize(cv.imread("../../INRIAPerson/Test/neg/00001241.png",-1),(64,128))
print("Obteniendo los descriptores de las imagenes pregunta")
desc = descriptorHOG.obtainDescriptors([img,img2])
print("Predicción: ")
print(svm.predict(desc))
