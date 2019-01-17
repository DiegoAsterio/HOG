import descriptorHOG
import auxFunc
import numpy as np
import cv2

################################################################################
##                Test de la corrección Gamma (Parte 1)                       ##
################################################################################

def test1():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    auxFunc.pintaMI([img,normalized])


################################################################################
##              Test del cómputo del gradiente (Parte 2)                      ##
################################################################################

def test2Paper():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    print(descriptorHOG.gradientComputation1DPaper(normalized,1))

def test2Alt1():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    print(descriptorHOG.gradientComputation1DAlt1(normalized,1))

def test2Alt2():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    print(descriptorHOG.gradientComputation1DAlt2(normalized,1))

def test2Alt3():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    print(descriptorHOG.gradientComputation1DAlt3(normalized,1))


################################################################################
##             Test del orientation binning (Parte 3)                         ##
################################################################################

def test3():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    normalized = descriptorHOG.gammaNormalization(img)
    dx, dy = descriptorHOG.gradientComputation1DPaper(normalized,1)
    histograms = descriptorHOG.spatialOrientationBinning(dx,dy)
    print(histograms)
    print("Es una matriz de tamaño: " + str(histograms.shape))


################################################################################
##        Test del la normalización y descriptor por bloques (Parte 4)        ##
################################################################################

def test4RHOG():
    img = cv2.imread("../../INRIAPerson/Test/pos/crop001501.png",-1)
    smoothed = descriptorHOG.rhog(img)
    print("Imagen original a la izquierda y suavizado R-HOG a la derecha")
    auxFunc.pintaMI([img,smoothed])

################################################################################
################################################################################

print("Quiere realizar todo el test o alguna función concreta [t/1/2/3/4]")
option = input()

if option=='1' or option=='t':
    print("############################################################")
    print("Test de la normalización gamma (Parte 1)")
    print("############################################################\n\n")
    test1()
if option=='2' or option=='t':
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
if option=='3' or option=='t':
    print("############################################################")
    print("Test del cómputo del spatial/orientation binning (Parte 3)")
    print("############################################################\n\n")
    test3()
if option=='4' or option=='t':
    print("############################################################")
    print("Test del suavizado R-HOG (Parte 4)")
    print("############################################################\n\n")
    test4RHOG()
if option != 't' and option!='1' and option!='2' and option!='3' and option!='4':
    print(option + " no es una opción válida")
