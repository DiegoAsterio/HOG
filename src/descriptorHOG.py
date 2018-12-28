import numpy as np
import cv2
import auxFunc


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
            histograms.append(auxFunc.computeHistogram(gradients[i:i+tam_cel,j:j+tam_cel]))

    return histograms
