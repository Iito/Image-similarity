import argparse
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import os
import pdb
import platform
from pprint import pprint
from re import I
from time import time
from typing import *
import sys

import cv2
import numpy as np
from tqdm import tqdm
import rawpy

from template_matching import imageSimilarityCPU, imageSimilarityCUDA

GREY_SCALE = cv2.COLOR_BGR2GRAY

def loadImg(img: str, grey: bool = True) -> np.ndarray:
	
	if img.lower().endswith(".nef"):
		r = rawpy.imread(img).postprocess()
	else:
		r = cv2.imread(img)
	if grey:
		r = cv2.cvtColor(r, GREY_SCALE)
	return r

def loadImgCUDA(img: str, grey: bool = True) -> cv2.cuda_GpuMat:
	
	if not isinstance(img, np.ndarray):
		img = loadImg(img, False)
	cuImg = cv2.cuda_GpuMat()
	cuImg.upload(img)
	if grey:
		cuImg = cv2.cuda.cvtColor(cuImg, GREY_SCALE) 
	return cuImg

def findSimilarity(dataset: Dict[str, Union[np.ndarray, cv2.cuda_GpuMat]], 
					imageSimilarity, p, thresh: int, match_ratio: float,
					batch_size: int) -> List[ List [str]]:

	nb_images = len(dataset)
	result = []
	prev_res, next_res = set(), set()
	list_img = np.array(list(dataset.keys()))
	imgs = np.array(list(dataset.values()))

	offset = batch_size
	for i in tqdm(range(0, nb_images-1)):
		current_img = list_img[i]
		template = imgs[i]
		end = i+offset
		nxt_img_name = np.array(list_img[i+1:end])
		to_pop = []

		if current_img in prev_res:
			for n, x in enumerate(nxt_img_name):
				if x in prev_res:
					to_pop.append(n)
		to_off = len(to_pop)

		if to_off > 0:

			nxt_img_name = np.delete(nxt_img_name, to_pop)
			nxt_img_name = np.array(nxt_img_name)

		next_images = imgs[np.isin(list_img, nxt_img_name)]
		part = partial(imageSimilarity, template, thresh=thresh, nn_match_ratio=match_ratio)
		results = p.map(part, next_images)

		results = np.array(results)
		if results.shape[0] > 0:
			names = nxt_img_name[results].tolist()
		else:
			names = nxt_img_name.tolist()

		if len(names) != 0:
	
			prev_res = set()
			[prev_res.add(x) for x in names]
			names.append(current_img)
			names.sort()
			result.append(names)
	results = []
	nb_groups = len(result)

	if nb_groups > 1:
		glob_set = set()
		for i in range(nb_groups-1):
			g1 = result[i]
			out = set([x for x in g1 if x not in glob_set])
			for j in range(i + 1, nb_groups):
				g2 = result[j]
				new = any([x in g2 for x in out])
				if new:
					out = out.union(set(g2))
			glob_set = glob_set.union(out)
			if out != set():
				results.append(list(out))
	else:
		results = result

	return results

def blurDetection(images: List[ Dict[ str, Union[ cv2.cuda_GpuMat, np.ndarray]]]) -> List[ Tuple[ str, float]]:
	"""
	Calculates image blur using the variance of laplacian method then orders the images from less blurr to most blurry
	:param images: A list of images name or a list of numpy array representing the image.
	:return List of tuples: [(img1.jpg, 1000), (img2.jpg, 650), (img3.jpg, 126)]
	"""
	scores = []
	for name in images.keys():
		image = images[name]
		if not isinstance(image, np.ndarray):
			img_grey = image.download()
		else:
			img_grey = image
		laplace = cv2.Laplacian(img_grey, cv2.CV_64F).var()
		scores.append((name, laplace))
	# Insure the first result of the list will be the best, the last, the worst.
	scores = sorted(scores, key=lambda scores: scores[1], reverse=True)
	return scores

def sortAndOrder(group: List[Tuple[str, float]], move: bool = False) -> None:
	"""
	This method process groups of images with similarity, keep the best one according to blur detection
	and move all the other in a folder named after the best one.
	It does not delete anything.
	It also writes in the newly created folder a text file containing the blur score of each picture within the group.
	"""
	for i, img_score in enumerate(group):
		image = img_score[0]
		score = img_score[1]
		path = image.split(os.path.sep)
		img_name = path[-1]
		path = os.path.sep.join(path[:-1])
		if i == 0:
			grp_name = img_name.split(".")[0]
			dir_name = os.path.join(path, grp_name)
			if not os.path.exists(dir_name) and move:
				os.makedirs(dir_name)
			fp = dir_name if move else path
			fn = open(os.path.join(fp, f"{grp_name}_scores.txt"), 'a')
			fn.write(f"{img_name}\t{score}\n")
			continue
		new_img_path = os.path.join(dir_name, img_name)
		if move:
			os.rename(image, new_img_path)
		fn.write(f"{img_name}\t{score}\n")


if __name__ == "__main__":
	param = argparse.ArgumentParser()

	param.add_argument("--dir", '-d', type=str, required=True)
	param.add_argument("--cuda", action='store_true', default=False)
	param.add_argument("--threshold", '-th', type=int, default=200)
	param.add_argument("--match_ratio", "-mr", type=float, default=0.8)
	param.add_argument("--blur", type=int, default=100)
	param.add_argument("--rename", action='store_true', default=False)
	param.add_argument("--batch_size", type=int, default=10)
	param = param.parse_args()

	CUDA = param.cuda
	IMG_EXT = (".jpg", ".jpeg")
	path = param.dir

	if platform.system() != "Windows":
		path = "/mnt/e" + path
	else:
		path = os.path.abspath(path)
	start_time = time()
	list_img = os.listdir(path)
	list_img = [os.path.join(path, x) for x in list_img if x.lower().endswith(IMG_EXT)]
	list_img.sort()
	if len(list_img) < 2:
		print("You need at least 2 images to  compare")
		sys.exit()
	tp = ThreadPool()
	pool = Pool()

	if CUDA:
		try:
			cv2.cuda.getDevice()
		except:
			print("No CUDA detected or opencv hasn't been compile with CUDA")
			CUDA = False

	# Load images
	print(f"Loading {len(list_img)} images into memory")
	GREY = False if CUDA else True
	
	loadImgs = partial(loadImg, grey=GREY)
	imgs = pool.map(loadImgs, list_img)

	if CUDA:
		pool = tp
		imgs = pool.map(loadImgCUDA, imgs)
		imageSimilarity = imageSimilarityCUDA
	else:
		imageSimilarity = imageSimilarityCPU

	
	dataset = {k:v for k, v in zip(list_img, imgs)}
	similar_imgs_name = findSimilarity(dataset, imageSimilarity, pool, param.threshold, param.match_ratio, param.batch_size)
	group = len(similar_imgs_name)
	nb_per_group = [len(x) for x in similar_imgs_name]

	print(f"There {'are' if group > 1 else 'is'} {group} group of similar images")
	
	similar_imgs = []

	for group in similar_imgs_name:
		group_array = {x:dataset[x] for x in group}
		similar_imgs.append(group_array)

	print("Getting the best image out of each group")
	bests = pool.map(blurDetection, similar_imgs)
	sortImages = partial(sortAndOrder, move=param.rename)
	tp.map(sortImages, bests)
	
	print(time() - start_time)


	