import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy import stats

def histogram(img):
	color = ('b','g','r')
	for i,col in enumerate(color):
		histr = cv2.calcHist([img],[i],None,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
	plt.show()

def get_area(img):
	return np.count_nonzero(img)

def get_mode_hue(hsv):
	nonzeros = hsv[hsv[:,:,0]>0]
	return stats.mode(nonzeros[nonzeros<100])
def get_stats(hsv):
	nonzeros = hsv[hsv[:,:,0]>0]
	mode = stats.mode(nonzeros[nonzeros<100])[0][0]
	gmean = stats.gmean(nonzeros[nonzeros<100])
	kurtosis = stats.kurtosis(nonzeros[nonzeros<100])
	var = stats.kstatvar(nonzeros[nonzeros<100])
	std = stats.tstd(nonzeros[nonzeros<100])
	return str(mode),str(gmean),str(kurtosis),str(var),str(std)
def get_min_max(mask,folder):
	xs,ys = np.where(mask>0)
	sorted_xs = sorted(xs)
	sorted_ys = sorted(ys)
	if folder == "2low_total_spikelets_high_seed_set":
		while ((sorted_xs[0] >= 860) and (sorted_xs[0] <= 863)):
			sorted_xs = sorted_xs[1:]
		while (sorted_ys[0] == 1616) or (sorted_ys[0] == 1617):
			sorted_ys = sorted_ys[1:]
	elif folder == "2high_exsertion_low_seed_set":
		while (sorted_xs[0] == 458) or (sorted_xs[0] == 459):
			sorted_xs = sorted_xs[1:]
		while (sorted_ys[0] == 1922) or (sorted_ys[0] == 1923):
			sorted_ys = sorted_ys[1:]

	min_x = sorted_xs[0]
	min_y = sorted_ys[0]
	max_x = sorted_xs[-1]
	max_y = sorted_ys[-1]
	return min_x,min_y,max_x,max_y

def get_blob_center(mask,img):
	M = cv2.moments(mask)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	img = cv2.circle(img, (cX, cY), 3, (0, 0, 255), -1)
	return img

def get_upper_quadrant(mask):
	h,w = mask.shape[:2]
	mask_upper = mask[:h//2,:]
	left = mask_upper[:,:w//3]
	right = mask_upper[:,(w//3)*2:]
	lscore = np.count_nonzero(left)
	rscore = np.count_nonzero(right)
	if lscore > rscore:
		return 0
	else:
		return 1

def get_orientation(mask,img,folder):
	h,w = img.shape[:2]
	img = cv2.copyMakeBorder(img, h//4, h//4, w//4, w//4, borderType=cv2.BORDER_CONSTANT)
	mask = cv2.copyMakeBorder(mask, h//4, h//4, w//4, w//4, borderType=cv2.BORDER_CONSTANT)
	h,w = img.shape[:2]
	M = cv2.moments(mask)
	quad = get_upper_quadrant(mask)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	phi = 0.
	if M["m20"] != M["m02"]:
		phi = (0.5*(np.arctan((2*M["m11"])/(M["m20"]-M["m02"]))))*(180./np.pi)
	if quad == 0:
		rotM = cv2.getRotationMatrix2D((cX,cY),phi,1.0)
	else:
		rotM = cv2.getRotationMatrix2D((cX,cY),-phi,1.0)
	img = cv2.warpAffine(img,rotM,dsize=(w,h))
	img_bin = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,img_bin = cv2.threshold(img_bin,50,255,cv2.THRESH_BINARY)
	min_x,min_y,max_x,max_y = get_min_max(img_bin,folder)
	img = img[min_x:max_x,min_y:max_y]
	h,w = img.shape[:2]
	ratio = w/h
	if ratio > 1:
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		h,w = img.shape[:2]
		ratio = w/h
	return img,ratio

names = ["2high_exsertion_low_seed_set","2low_seed_set_low_exsertion","2low_total_spikelets_high_seed_set"]
# names = ["2low_seed_set_low_exsertion","2low_total_spikelets_high_seed_set"]
mask_dir = "../images/mask_set/"
img_dir = "../preprocessed/"
out_dir = "../images/mask_applied/"
roi_out_dir = "../images/cropped_rois/"
roi_moment_dir = "../images/cropped_rois_with_center/"
csv = "stats.csv"

filewrite = "img,folder,ratio,h,w,h_cm,w_cm,area,mode,gmean,kurtosis,var,std\n"
for i in range(len(names)):
	mask_set_dir = mask_dir+names[i]
	img = cv2.imread(img_dir+names[i]+".jpg")
	main = np.zeros(img.shape)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_shape = img.shape
	main_fname = str(i)+"_COMBINED.jpg"
	for j in glob.glob(mask_set_dir+"/*"):
		print(j)
		folder = j.split("/")[-2]
		curr_mask = cv2.imread(j,0)
		ret,curr_mask = cv2.threshold(curr_mask,100,255,cv2.THRESH_BINARY)
		min_x,min_y,max_x,max_y = get_min_max(curr_mask,folder)
		res = cv2.bitwise_and(img,img, mask= curr_mask)
		roi_colored = res[min_x:max_x,min_y:max_y]
		hsv_res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
		mode_color = cv2.cvtColor(np.uint8([[[get_mode_hue(hsv_res)[0],200,200]]]),cv2.COLOR_HSV2BGR)[0][0]
		overlay = np.full(img_shape,mode_color)
		res_new = cv2.bitwise_and(overlay,overlay, mask= curr_mask)
		mode,gmean,kurtosis,var,std = get_stats(hsv_res)
		# main += res_new
		aligned,ratio = get_orientation(curr_mask[min_x:max_x,min_y:max_y],roi_colored,folder)
		# cv2.imwrite(out_dir+names[i]+"/"+j.split("/")[-1],res)
		# cv2.imwrite(out_dir.replace("mask_applied","mask_applied_ave")+names[i]+"/"+j.split("/")[-1],res_new)
		# cv2.imwrite(roi_out_dir+names[i]+"/"+j.split("/")[-1],roi_colored)
		# cv2.imwrite(roi_out_dir.replace("rois","rois_bw")+names[i]+"/"+j.split("/")[-1],curr_mask[min_x:max_x,min_y:max_y])
		# cv2.imwrite(roi_moment_dir+names[i]+"/"+j.split("/")[-1],aligned)
		# cv2.imwrite(main_fname,main)
		filewrite+=(",".join([j,folder,str(ratio),str(aligned.shape[0]),str(aligned.shape[1]),str(aligned.shape[0]/95),str(aligned.shape[1]/95),str(get_area(roi_colored)),mode,gmean,kurtosis,var,std])+"\n")

with open(csv,"w") as outfile:
	outfile.write(filewrite)
# overlay = np.zeros(reduce((lambda x,y:x*y),img_shape)).reshape(img_shape)
# while(sorted_xs[-1]-sorted_xs[0]> 150) or (sorted_xs[0] == 458) or (sorted_xs[0] == 459):
	# 	sorted_xs = sorted_xs[1:]
	# while(sorted_ys[-1]-sorted_ys[0]> 150) or (sorted_ys[0] == 1922) or (sorted_ys[0] == 1923):
	# 	sorted_ys = sorted_ys[1:]