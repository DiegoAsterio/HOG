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
    '''
    @brief Funcion que computa el gradiente que en el paper funciona mejor
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,0,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,0,1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt1(img,sigma):
    '''
    @brief Funcion que computa el gradiente que en el paper funciona mejor
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt2(img,sigma):
    '''
    @brief Funcion que computa el gradiente que en el paper funciona mejor
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([1,-8,0,8,-1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([1,-8,0,8,-1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt3(img,sigma):
    '''
    @brief Funcion que computa el gradiente que en el paper funciona mejor
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
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
        row_histograms = []
        for j in range(0,cols,tam_cel):
            # Añade el histograma de la célula
            row_histograms.append(af.computeHistogram(gradients[i:i+tam_cel,j:j+tam_cel]))
        histograms.append(row_histograms)

    return np.array(histograms)

################################################################################
##                 4: Normalization and Descriptor Blocks                     ##
################################################################################

def rhog(img,tam_bloque=2,tam_cel=3):
    '''
    @brief Función que divide la imagen img en bloques y aplica una gaussiana a
    los mismos con sigma=0.5*tam_bloque
    @param img Imagen a la que queremos aplicar la gaussiana localmente
    @param tam_bloque Número de celdas por lado del bloque
    @param tam_cel Número de píxeles por celda
    @return Devuelve una imagen en la que se ha aplicado una gaussiana a cada
    submatriz contenida en un bloque
    '''
    # Creamos una copia de la imagen
    img_aux = np.copy(img)
    # Sigma especificado en el paper
    sigma = tam_bloque*0.5
    # Tamaño en píxeles del bloque por lado
    size_block = tam_cel*tam_bloque
    for i in range(0,img_aux.shape[0],size_block):
        for j in range(0,img_aux.shape[1],size_block):
            # Obtenemos el suavizado en la submatriz
            local_gauss = cv.GaussianBlur(img_aux[i:i+size_block,j:j+size_block])
            # Modificamos los valores de la imagen auxiliar con los de la gaussiana
            img_aux = af.modifyLocalMatrix(img_aux,local_gauss,i,i+size_block,j,j+size_block)
    return img_aux
