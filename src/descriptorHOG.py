import numpy as np
import cv2
import auxFunc as af

################################################################################
##                        1: Normalización Gamma                              ##
################################################################################

def gammaNormalization(img,c1=1,c2=0.5):
    '''
    @brief Función que corrige el valor gamma usando la ley de potencias con
    coeficientes c1 y c2.
    @param img Imagen a la que queremos normalizarle el valor gamma.
    @param c1 Factor multiplicativo en la normalización, por defecto 1.
    @param c2 Exponente de la normalización, por defecto 1/2 (compressing normalization).
    '''
    reduced = img/255.0
    corrected = np.power(reduced*c1,c2)
    return (corrected*255).astype(np.uint8)

def gradientComputation1DPaper(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,None,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,0,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,0,1],False,imgAux)
    return getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt1(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,None,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,1],False,imgAux)
    return getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt2(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,None,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([1,-8,0,8,-1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([1,-8,0,8,-1],False,imgAux)
    return getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt3(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,None,(0,0),sigma)
    mask1 = np.array([[0,-1],[1,0]])
    outputSignalsdx = cv.filter2D(img,-1,mask1)
    mask2 = np.array([[-1,0],[0,1]])
    outputSignalsdy = cv.filter2D(img,-1,mask2)

    return getGradient(outputSignalsdx, outputSignalsdy)
