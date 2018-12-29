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

test2Paper()
