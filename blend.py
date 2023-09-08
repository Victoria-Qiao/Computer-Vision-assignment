import cv2
import numpy as np
import PIL
from matches import Match
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter
import scipy.ndimage.filters
from scipy.signal import convolve
from skimage.transform import resize
from scipy import interpolate


class Blend:
	def __init__(self):
		return

	def reduce(self,im):
		return im[:2, :2]

	def blend(self, img1, img2,n, mask1, mask2):
		# plt.imshow(img1),plt.show()

		i1 = img1.copy()
		i2 = img2.copy()

		imgPyramid1 = [i1]
		imgPyramid2 = [i2]

		mask11 = mask1.astype('uint8')
		mask22 = mask2.astype('uint8')
		mask1Pyramid = [mask11]
		mask2Pyramid = [mask22]

		#build gaussian pyramid of the two images
		#and mask1/mask2 for the two images
		for i in range(0,n-1):
			i1 = cv2.pyrDown(i1)
			i2 = cv2.pyrDown(i2)
			# plt.imshow(i1),plt.show()
			# m1 = np.zeros_like(i1)
			# m2 = np.zeros_like(i2)
			# print(m1.shape)

			# m1[:,:] = mask11[::2,::2]
			# m2[:,:] = mask22[::2,::2]

			# plt.imshow(m1),plt.show()
			# print(m1)
			# mask11=m1
			# mask22=m2
			# mask11 = cv2.resize(mask11,(i1.shape[1],i1.shape[0]),interpolation=cv2.INTER_NEAREST)
			# plt.imshow(mask11),plt.show()
			# mask22 = cv2.resize(mask22,(i1.shape[1],i1.shape[0]),interpolation=cv2.INTER_NEAREST)
			# print(mask11)
			mask11 = cv2.pyrDown(mask11)
			mask22 = cv2.pyrDown(mask22)
			# print(np.max(mask11))
			# plt.imshow(mask33),plt.show()
			imgPyramid1.append(i1)
			imgPyramid2.append(i2)
			mask1Pyramid.append(mask11)
			mask2Pyramid.append(mask22)


		#build laplacian pyramid for the two images
		#from their gaussian pyramid
		laplacePyramid1 = [imgPyramid1[n-1]]
		laplacePyramid2 = [imgPyramid2[n-1]]

		for i in range(n-1, 0, -1):
			up1 = cv2.pyrUp(imgPyramid1[i])
			up2 = cv2.pyrUp(imgPyramid2[i])

			if (imgPyramid1[i-1].shape[0] < up1.shape[0]):
				up1 = np.delete(up1,-1,axis=0)
				up2 = np.delete(up2,-1,axis=0)

			if (imgPyramid1[i-1].shape[1] < up1.shape[1]):
				up1 = np.delete(up1,-1,axis=1)
				up2 = np.delete(up2,-1,axis=1)
	
			laplacePyramid1.append(cv2.subtract(imgPyramid1[i-1],up1))
			laplacePyramid2.append(cv2.subtract(imgPyramid2[i-1],up2))

		mP1 = []
		mP2 = []
		for i in range(0,n):
			mP1.append(mask1Pyramid[n-1-i])
			mP2.append(mask2Pyramid[n-1-i])

		mask1Pyramid = mP1
		mask2Pyramid = mP2

		Lout = []

		#use mask gaussian pyramid to multiply laplacian pyramid for each level and add together
		#to get blending maps for each level
		Lout =  np.multiply(mask1Pyramid, laplacePyramid1) + np.multiply(np.subtract(1,mask1Pyramid), laplacePyramid2)

		img = Lout[0]

		# Lout1 =  np.multiply(mask1Pyramid, laplacePyramid1)
		# Lout2 =  np.multiply(mask2Pyramid, laplacePyramid2)
		# img1 = Lout1[0]
		# img2 = Lout2[0]
		#from the top level to do upsampling and addtion to the lower level until the final level
		#to get the final blended image
		for i in range(1,n):
			img = cv2.pyrUp(img)

			if (img.shape[0] > Lout[i].shape[0]):
				img = np.delete(img,-1,axis=0)

			if (img.shape[1] > Lout[i].shape[1]):
				img = np.delete(img,-1,axis=1)


			img = cv2.add(img,Lout[i])
		return img
