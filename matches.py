import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL
import math
from mpl_toolkits.mplot3d import axes3d


MIN_MATCH_COUNT = 15

class Match:
	def __init__(self):
		self.sift = cv2.xfeatures2d.SIFT_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=10)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def siftFeatures(self, im):
		#compute sift feature points for the image
		kp, des = self.sift.detectAndCompute(im, None)
		return {'kp':kp, 'des':des}

	def match(self, i1, i2, direction=None):
		#find sift points for the two images
		imageSet1 = self.siftFeatures(i1)
		imageSet2 = self.siftFeatures(i2)
		matches = self.flann.knnMatch(
			imageSet1['des'],
			imageSet2['des'],
			k=2
			)
		good = []
		#choose good matches
		for i, (m,n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				good.append(m)

		if len(good) > MIN_MATCH_COUNT:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']

			#if good matches is greater than a threshold, then it could be seen
			#as the two images are overlapping and homography is calculated
			matchedPointsCurrent = np.float32(
				[pointsPrevious[m.queryIdx].pt for m in good]
			    ).reshape(-1,1,2)
			matchedPointsPrev = np.float32(
				[pointsCurrent[m.trainIdx].pt for m in good]
				).reshape(-1,1,2)

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			return H, matchedPointsCurrent, matchedPointsPrev
		return None



