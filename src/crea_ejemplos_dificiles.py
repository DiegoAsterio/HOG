import descriptorHOG
import auxFunc as af


print("Obteniendo los datos de entrenamiento")
# Obtenemos los datos de entrenamiento
td = descriptorHOG.obtainTrainData()
print("Entrenando la SVM")
# Entrenamos la SVM
svm = descriptorHOG.trainSVM(td)
print("Obteniendo los ejemplos dificiles")
af.obtainHardExamples(svm)
