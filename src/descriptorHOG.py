import numpy as np
import cv2 as cv
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
    # Restringe los valores de la imagen entre 0 y 1
    reduced = img/255.0
    # Hace la correción de la luz
    corrected = np.power(reduced*c1,c2)
    return (corrected*255).astype(np.uint8)

################################################################################
##                        2: Cómputo del gradiente                            ##
################################################################################

def gradientComputation1DPaper(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,0,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,0,1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt1(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt2(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([1,-8,0,8,-1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([1,-8,0,8,-1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt3(img,sigma):
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    mask1 = np.array([[0,-1],[1,0]])
    outputSignalsdx = cv.filter2D(img,-1,mask1)
    mask2 = np.array([[-1,0],[0,1]])
    outputSignalsdy = cv.filter2D(img,-1,mask2)

    return af.getGradient(outputSignalsdx, outputSignalsdy)

################################################################################
##                      3: Spatial/Orientation Binning                        ##
################################################################################

def spatialOrientationBinning(gradients,tam_cel=3):
    '''
    @brief Función que dada una matriz de gradientes y un tamaño de célula divide la matriz en
    células, calcula los histogramas de todas y los devuelve en un vector.
    @param gradients Matriz con los gradientes
    @param tam_cel Tamaño de la célula, por defecto 3.
    '''
    # Obtiene el número de filas y columnas de la imagen
    rows = gradients.shape[0]
    cols = gradients.shape[1]

    # Inicializa los histogramas
    histograms = []

    # Divide la matriz en celdas y llama con cada una al cálculo de histogramas.
    for i in range(0,rows,tam_cel):
        for j in range(0,cols,tam_cel):
            # Añade el histograma de la célula
            histograms.append(af.computeHistogram(gradients[i:i+tam_cel,j:j+tam_cel]))

    return histograms
