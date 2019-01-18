import descriptorHOG
import auxFunc as af
import cv2 as cv

print("Obtener los ejemplos dificiles positivos, negativos o todos [p/n/t]")
option = input()
print("Cargar la svm de fichero o entrenarla de nuevo [c/e]")
option_load = input()

if option=="n" or option=="t":
    if option_load=="e":
        print("Obteniendo los datos de entrenamiento")
        # Obtenemos los datos de entrenamiento
        td = descriptorHOG.obtainTrainData()
        print("Entrenando la SVM")
        # Entrenamos la SVM
        svm = descriptorHOG.trainSVM(td)
        del td
    if option_load=="c":
        print("Cargando la SVM de fichero")
        svm = cv.ml.SVM_load("svm.txt")
    print("Obteniendo los ejemplos dificiles negativos")
    af.obtainHardNegativeExamples(svm)

if option=="p" or option=="t":
    if option_load=="e":
        print("Obteniendo los datos de entrenamiento")
        # Obtenemos los datos de entrenamiento
        td = descriptorHOG.obtainTrainData()
        print("Entrenando la SVM")
        # Entrenamos la SVM
        svm = descriptorHOG.trainSVM(td)
        del td
    if option_load=="c":
        print("Cargando la SVM de fichero")
        svm = cv.ml.SVM_load("svm.txt")
    print("Obteniendo los ejemplos dificiles positivos")
    af.obtainHardPositiveExamples(svm)
