import descriptorHOG
import auxFunc
import numpy as np
import cv2

################################################################################
##                         Auxiliar de testing                                ##
################################################################################

def pintaMI(vim):
    imagenes = []
    max_h = 0
    max_w = 0
    for im in vim:
        if im.shape[0]>max_h:
            max_h = im.shape[0]
        if im.shape[1]>max_w:
            max_w = im.shape[1]
    for im in vim:
        if len(im.shape)==2:
            imagenes.append(cv2.copyMakeBorder(cv2.cvtColor(im,cv2.COLOR_GRAY2RGB),top=0,bottom=max_h-im.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=[0,0,0]))
        else:
            imagenes.append(cv2.copyMakeBorder(im,top=0,bottom=max_h-im.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=[0,0,0]))
    concatenada = cv2.hconcat(imagenes)
    cv2.namedWindow('Imagenes', cv2.WINDOW_NORMAL)
    cv2.imshow("Imagenes",concatenada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################################################################################
##                Test de la corrección Gamma (Parte 1)                       ##
################################################################################

def test1():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    pintaMI([img,normalized])


################################################################################
##              Test del cómputo del gradiente (Parte 2)                      ##
################################################################################

def test2Paper():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    print(descriptorHOG.gradientComputation1DPaper(img,1))

def test2Alt1():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    print(descriptorHOG.gradientComputation1DAlt1(img,1))

def test2Alt2():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    print(descriptorHOG.gradientComputation1DAlt2(img,1))

def test2Alt3():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    print(descriptorHOG.gradientComputation1DAlt3(img,1))


################################################################################
##             Test del orientation binning (Parte 3)                         ##
################################################################################

def test3():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    gradients = descriptorHOG.gradientComputation1DPaper(img,1)
    histograms = descriptorHOG.spatialOrientationBinning(gradients)
    print(histograms)
    print("Es una matriz de tamaño: " + str(histograms.shape))


################################################################################
################################################################################

print("Quiere realizar todo el test o alguna función concreta [t/1/2/3]")
option = input()

if option=='1' or option=='t':
    print("############################################################")
    print("Test de la normalización gamma (Parte 1)")
    print("############################################################\n\n")
    test1()
elif option=='2' or option=='t':
    print("############################################################")
    print("Test del cómputo del gradiente según el paper (Parte 2)")
    print("############################################################\n\n")
    test2Paper()

    print("############################################################")
    print("Test del cómputo del gradiente alternativa 1 (Parte 2)")
    print("############################################################\n\n")
    test2Alt1()

    print("############################################################")
    print("Test del cómputo del gradiente alternativa 2 (Parte 2)")
    print("############################################################\n\n")
    test2Alt2()

    print("############################################################")
    print("Test del cómputo del gradiente alternativa 3 (Parte 2)")
    print("############################################################\n\n")
    test2Alt3()
elif option=='3' or option=='t':
    print("############################################################")
    print("Test del cómputo del spatial/orientation binning (Parte 3)")
    print("############################################################\n\n")
    test3()
else:
    print(option + " no es una opción válida")
