import myAuxFuncs as da
import auxFunc as af
import cv2 as cv

if __name__ == "__main__":
    im = da.leeImagen("./plane.bmp", cv.IMREAD_COLOR)
    outputSignalsdx = af.convoluteWith1DMask([-1,0,1],True,im)
    outputSignalsdy = af.convoluteWith1DMask([-1,0,1],False,im)
    im2 = da.leeImagen("./plane.bmp", cv.IMREAD_GRAYSCALE)
    outputSignalsGSdx = af.convoluteWith1DMask([-1,0,1],True,im2)
    print(outputSignalsGSdx)
    da.pintaI(outputSignalsGSdx)
