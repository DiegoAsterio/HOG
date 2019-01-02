import numpy as np
import cv2 as cv
import auxFunc as af
import pdb

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
    @brief Funcion que computa el gradiente utilizando la mascara 1D [-1,0,1]
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
    @brief Funcion que computa el gradiente usando la mascara [-1,1]
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
    @brief Funcion que computa el gradiente usando la mascara [1,-8,0,8,-1]
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
    @brief Funcion que computa el gradiente usando una mascara 2D
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

def spatialOrientationBinning(gradients,tam_cel=3,num_cols=9):
    '''
    @brief Función que dada una matriz de gradientes y un tamaño de célula divide la matriz en
    células, calcula los histogramas de todas y los devuelve en un vector.
    @param gradients Matriz con los gradientes
    @param tam_cel Tamaño de la célula, por defecto 3.
    @param num_cols Numero de columnas del histograma, por defecto 9.
    '''
    # Obtiene el número de filas y columnas de la imagen
    rows = gradients.shape[0]
    cols = gradients.shape[1]

    # Inicializa los histogramas
    histograms = []
    nrow = 0
    ncol = 0
    contar = True

    # Divide la matriz en celdas y llama con cada una al cálculo de histogramas.
    for i in range(0,rows,tam_cel):
        row_histograms = []
        for j in range(0,cols,tam_cel):
            if i+tam_cel<rows and j+tam_cel<cols:
                ncol = ncol+1 if contar else ncol
                # Añade el histograma de la célula
                row_histograms.append(af.computeHistogramDiego(gradients[i:i+tam_cel,j:j+tam_cel],num_cols))
        histograms.append(row_histograms)
        nrow+=1
        contar=False
    pdb.set_trace()
    return np.array(histograms).reshape((nrow,ncol,num_cols))

################################################################################
##                 4: Normalization and Descriptor Blocks                     ##
################################################################################

def normalizeDescriptor(bloque):
    ret = bloque.reshape(-1)
    ret = np.array(list(map(lambda x : x/af.normaEuclidea(x),ret)))
    return ret

def rhog(histogramas,tam_bloque=3):
    '''
    @brief Función que calcula los descriptores normalizados a partir de los
    histogramas de cada celula dentro de un mismo bloque
    @param histogramas Todos los histogramas computados a partir de celulas
    @param tam_bloque Tamano del bloque debe ser una pareja e.g. (2,2)
    @return Devuelve un array que separa en bloques los histogramas
    '''
    pdb.set_trace()
    n, m, k = histogramas.shape
    descriptores = []
    for i in range(n)-tam_bloque[0]:
        descriptoresFila = []
        for j in range(m)-tam_bloque[1]:
            descriptor = normalizeDescriptor(histogramas[i:i+tam_bloque[0]][j:j+tam_bloque[1]])
            descriptoresFila.append(descriptor)
        descriptores.append(descriptoresFila)
    return np.array(descriptores)

#def normalizechog(subseccion, radio_central, num_secciones, expansion):

def chog(histogramas, radio_central, num_secciones, expansion):
    n, m, k = histogramas.shape
    descriptores = []
    R = radio_central*(1+expansion)
    for i in range(R,n-R):
        descriptoresFila = []
        for j in range(R,m-R):
            descriptor = normalizechog(histogramas[i-R:i+R][j-R:j+R],radio_central, num_secciones, ratio)
            descriptoresFila.append(descriptor)
        descriptores.append(descriptoresFila)
    return np.array(descriptores)


################################################################################
##                           5: Classification                                ##
################################################################################


def trainSVM(trainData):
    svm = cv.ml.SVM_create()
    svm.setC(0.01)
    svm.train(trainingData)
    return svm

def testSVM(svm, testData):
    retval, results = svm.predict(testData)
    return results

def obtainTrainData():
    print("Cargando imágenes")
    imgs_pos,img_neg = af.loadTrainImgs()
    resp = np.concatenate((np.ones(len(imgs_pos[:1])),-np.ones(len(img_neg[:1]))))
    imgs = imgs_pos[:1]+img_neg[:1]
    print("Normalización Gamma")
    gamma_corrected = []
    for im in imgs:
        gamma_corrected.append(gammaNormalization(im))
    print("Calculando los gradientes")
    gradients = []
    for gam in gamma_corrected:
        gradients.append(gradientComputation1DPaper(gam,1))
    print("Calculando los histogramas")
    histograms = []
    for gra in gradients:
        histograms.append(spatialOrientationBinning(gra))
    print("Calculando los descriptores de imagen")
    img_descr = []
    for histo in histograms:
        img_descr.append(rhog(histo).reshape(-1))
    return cv.ml.TrainData_create(img_descr,cv.ml.ROW_SAMPLE,resp)
