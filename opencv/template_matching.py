import pdb
from math import sqrt

import cv2 as cv

GREY_SCALE = cv.COLOR_BGR2GRAY

def imageSimilarityCPU(img1, img2, thresh: int = 200, nn_match_ratio: float = 0.8):

	# Initiate ORB detector
	orb = cv.ORB_create(10000)

	# find the keypoints and descriptors with ORB
	kp1  = orb.detect(img1,None)
	kp2  = orb.detect(img2,None)

	descriptor = cv.xfeatures2d.BEBLID_create(0.75)
	kpts1, desc1 = descriptor.compute(img1, kp1)
	kpts2, desc2 = descriptor.compute(img2, kp2)

	# create BFMatcher object

	matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

	nn_matches_ = matcher.knnMatch(desc1, desc2, 2)
	nn_matches = [x for x in nn_matches_ if len(x) == 2]
	
	matched1 = []
	matched2 = []
	
	for m, n in nn_matches:
	    if m.distance < nn_match_ratio * n.distance:
	        matched1.append(kpts1[m.queryIdx])
	        matched2.append(kpts2[m.trainIdx])
	if len(matched1) >= thresh:
		return True
	else:
		return False

def imageSimilarityCUDA(cuMat1g, cuMat2g, thresh: int = 200, nn_match_ratio: float = 0.8):
	
	# Initiate ORB detector
	orb = cv.cuda_ORB.create(10000)

	# find the keypoints and descriptors with ORB
	kp1, desc1  = orb.detectAndComputeAsync(cuMat1g, None)
	kp2, desc2  = orb.detectAndComputeAsync(cuMat2g, None)
	kpts1   = orb.convert(kp1)
	kpts2   = orb.convert(kp2)

	desc1_ = cuMat1g.download()
	desc2_ = cuMat2g.download()

	# CPU descriptor
	descriptor = cv.xfeatures2d.BEBLID_create(0.80)
	kpts1, _desc1 = descriptor.compute(desc1_, kpts1)
	kpts2, _desc2 = descriptor.compute(desc2_, kpts2)

	
	# Upload BEBLIB descriptors to GpuMat
	desc1 = cv.cuda_GpuMat()
	desc2 = cv.cuda_GpuMat()
	desc1.upload(_desc1)
	desc2.upload(_desc2)

	# create BFMatcher object
	matcher = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_HAMMING)

	nn_matches = matcher.knnMatch(desc1, desc2, 2)
	#nn_matches = [x for x in nn_matches if len(x) == 2]

	matched1 = []
	matched2 = []
	for m, n in nn_matches:
	    if m.distance < nn_match_ratio * n.distance:
	        matched1.append(kpts1[m.queryIdx])
	        matched2.append(kpts2[m.trainIdx])
	
	if len(matched1) >= thresh:
		return True
	else:
		return False

if __name__ == "__main__":
	from time import time
	img1 =  'image_1.jpg'
	img2 =  'image_2.jpg'
	
	im1 = cv.imread(img1)
	im2 = cv.imread(img2)

	cuImg1 = cv.cuda_GpuMat()
	cuImg2 = cv.cuda_GpuMat()

	cuImg1.upload(im1)
	cuImg2.upload(im2)
	cuMat1g = cv.cuda.cvtColor(cuImg1, GREY_SCALE)
	cuMat2g = cv.cuda.cvtColor(cuImg2, GREY_SCALE)

	im1g = cv.cvtColor(im1, GREY_SCALE)
	im2g = cv.cvtColor(im2, GREY_SCALE)
	a = imageSimilarityCUDA(cuMat1g, cuMat2g)
	print(a)

