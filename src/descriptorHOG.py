import numpy as np
import cv2


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
