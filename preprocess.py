import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

gridsize = 10
# clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(gridsize,gridsize))
def gray_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def max_white(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))

def gimp(img, perc = 0.05):
    for channel in range(img.shape[2]):
        mi, ma = (np.percentile(img[:,:,channel], perc), np.percentile(img[:,:,channel],100.0-perc))
        img[:,:,channel] = np.uint8(np.clip((img[:,:,channel]-mi)*255.0/(ma-mi), 0, 255))
    return img

def histogram(img):
	color = ('b','g','r')
	for i,col in enumerate(color):
		histr = cv2.calcHist([img],[i],None,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
	plt.show()

outdir = "../preprocessed/"
masks = "../images/mask_tif/"
filenames = glob.glob("../images/orig/*.jpg")
for i in filenames:
	filename = i.split("/")[-1]
	masktif = cv2.imread(masks+filename.replace("jpg","tif"),0)
	ret, masktif = cv2.threshold(masktif, 254, 255, cv2.THRESH_BINARY)
	img = cv2.imread(i)
	# img = gray_world(img)
	# img = gimp(img,0.1)
	res = cv2.bitwise_and(img,img, mask= masktif)
	cv2.imwrite(outdir+filename,res)