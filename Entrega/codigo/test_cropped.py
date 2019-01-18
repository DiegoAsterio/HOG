import auxFunc as af
import descriptorHOG
import cv2 as cv

print("Cargando la SVM de fichero")
svm = cv.ml.SVM_load("svm_sin_hardpositives.txt")

print("Cargando el test")
img_pos, img_neg = af.loadTestImgs()

descriptors = descriptorHOG.obtainDescriptors(img_pos+img_neg)

predictions_raw = svm.predict(descriptors)[1]
del svm
predictions = [pred[0] for pred in predictions_raw]

pred_pos = predictions[:len(img_pos)]
pred_neg = predictions[len(img_pos):]

pos_score = 0
neg_score = 0

for pred in pred_pos:
    if pred==1:
        pos_score+=1

for pred in pred_neg:
    if pred==2:
        neg_score+=1

print("\n\n##################################################")
print("Positivos: " + str(pos_score) + "/" + str(len(pred_pos)) + "===>" + str(100*pos_score/len(pred_pos)) + "%")
print("Negativos: " + str(neg_score) + "/" + str(len(pred_neg)) + "===>" + str(100*neg_score/len(pred_neg)) + "%")
print("Porcentaje de acierto total: " + str(100*(pos_score+neg_score)/(len(pred_pos)+len(pred_neg))))
print("##################################################\n\n")
