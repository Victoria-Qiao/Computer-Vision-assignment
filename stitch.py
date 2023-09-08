import cv2
import numpy as np
import PIL
from matches import Match
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
from blend import Blend
import skimage.exposure
from skimage import data,img_as_float
from scipy import misc
import sys
import os

class Stitch:
	def __init__(self, args, ifcyl, ifresize,width,height):

		#if use cylindrical warping
		self.cylindrical = ifcyl
		#read paths from file
		self.path = args
		fp = open(self.path, 'r')
		self.filenames = [each.rstrip('\r\n') for each in fp.readlines()]
		#if the number of paths read in is less than or equal to 2, then
		#the program will show warning and exit
		if(len(self.filenames) <= 1):
			print('Minimum number of input images required: 2.')
			exit()

		#if the number of paths is greater than or equal to 2, then
		#read in images using cv2.imread. If the 'ifresize' option is
		#1, then the images will be read in and resized into (500,400)
		self.images = None
		if (ifresize == 1):
			self.images = [cv2.resize(cv2.imread(each),(height,width)) for each in self.filenames]
		else:
			self.images = [cv2.imread(each) for each in self.filenames]

		#Because in opencv, the images are in BGR format, the images will be transformed into
		#RGB which is convenient for matplotlib to show the images normally
		self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images]
		self.count = len(self.images)

		#set up match object for calculate matching points and homography for two images which 
		#have overlapping area.
		self.matcher_obj = Match()

		#set up blend object for blending two images using pyramid blending
		self.blend = Blend()

		#left array for recording images on the left part of panorama and right
		#array for recording images on the right part of panorama
		self.left = []
		self.right = []

		#self.center records the input images in center. If the number of input images is odd, then
		#the final outcome is the image in the middel else if the number of input images is even, then
		#the center image is set to None

		#limg record the final stitched image in left part images of the input images and rimg record the 
		#final stitched image in right part images of the input images
		self.center = None
		self.limg = None
		self.rimg = None

	def lrList(self):
		#get right and left image list and get center image from input images
		if (len(self.filenames) % 2 == 0):
			for i in range(len(self.filenames)):
				if (i < len(self.filenames)/2):
					self.left.append(self.images[i])
				else:
					self.right.append(self.images[i])

		else:
			self.center = self.images[int(len(self.filenames)/2)]
			for i in range(0,len(self.filenames)):
				if (i <= int(len(self.filenames)/2)-1):
					self.left.append(self.images[i])
				elif (i>int(len(self.filenames)/2)):
					self.right.append(self.images[i])

	def leftStitch(self):
		#focal length for cylindrical warping
		f = 500

		if (len(self.left) % 2 == 0):
			#stitching for even number of images
			imgs = []
			mask = []

			#first image stitching is done for every to images in the image array 
			#then these stitched images are combined together to form the stitched
			#image in the left part of images
			for i in range(0,len(self.left),2):
				mask = None

				img1 = self.left[int(i)]
				img2 = self.left[int(i+1)]
				#if cylindrical warping is set, then use cylindrical warping otherwise
				#use original images to do stitching
				if (self.cylindrical == 1):
					h,w,c = img2.shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img1 = self.cylindricalWarp(img1, K)
					img2 = self.cylindricalWarp(img2, K)
					# plt.imshow(img1),plt.show()
					# plt.imshow(img2),plt.show()
				img,msk = self.stitchimg(img1,img2,mask)
				imgs.append(img)

			img = None
			if (len(imgs) > 1):
				img,mask = self.stitchimg(imgs[len(imgs)-2],imgs[len(imgs)-1],None)
				for i in range(len(imgs) - 3,-1,-1):
					img,mask = self.stitchimg(imgs[i],img,mask)
				self.limg = img
			else:
				if (self.cylindrical == 1):
					img = imgs[0]
					h,w,c = img.shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img = self.cylindricalWarp(img, K)
					self.limg = img
				else:
					self.limg = imgs[0]
		else:
			imgs = []
			for i in range(0,len(self.left)-1,2):
				mask = None

				img1 = self.left[int(i)]
				img2 = self.left[int(i+1)]
				if (self.cylindrical == 1):
					h,w,c = img2.shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img1 = self.cylindricalWarp(img1, K)
					img2 = self.cylindricalWarp(img2, K)

				img,msk = self.stitchimg(img1,img2,mask)
				imgs.append(img)
			if (self.cylindrical == 1):
				h,w,c = self.left[-1].shape
				K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
				img = self.cylindricalWarp(self.left[-1],K)
				imgs.append(img)
			else:
				imgs.append(self.left[-1])

			if (len(imgs)>1):
				img,mask = self.stitchimg(imgs[len(imgs)-2],imgs[len(imgs)-1],None)

				for i in range(len(imgs) - 3,-1,-1):
					img,mask = self.stitchimg(imgs[i],img,mask)
				self.limg = img
			else:
				if (self.cylindrical == 1):
					h,w,c = self.left[0].shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img = self.cylindricalWarp(self.left[0], K)
					self.limg = img
				else:
					if (self.cylindrical == 1):
						self.limg = self.left[0]
						h,w,c = self.limg.shape
						K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
						self.limg = self.cylindricalWarp(self.limg, K)
					else:
						self.limg = self.left[0]



	def rightStitch(self):
		f = 500
		if (len(self.right) % 2 == 0):
			imgs = []
			#first image stitching is done for every to images in the image array 
			#then these stitched images are combined together to form the stitched
			#image in the right part of images
			for i in range(0,len(self.right),2):
				mask = None

				img1 = self.right[int(i+1)]
				img2 = self.right[int(i)]

				#if cylindrical warping is set, then use cylindrical warping otherwise
				#use original images to do stitching
				if (self.cylindrical == 1):
					h,w,c = img2.shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img1 = self.cylindricalWarp(img1, K)
					img2 = self.cylindricalWarp(img2, K)

				img,msk = self.stitchimg(img1,img2,mask)
				imgs.append(img)			
				img = None


			if (len(imgs)>1):
				img,mask = self.stitchimg(imgs[0],imgs[1],None)
				for i in range(2,len(imgs),1):
					img,mask = self.stitchimg(imgs[i],img,mask)
				self.rimg = img
			else:
				img = imgs[0]

			self.rimg = img
		else:
			imgs = []
			for i in range(0,len(self.right)-2,2):
				mask = None

				img1 = self.right[int(i+1)]
				img2 = self.right[int(i)]
				if (self.cylindrical == 1):
					h,w,c = img2.shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img1 = self.cylindricalWarp(img1, K)
					img2 = self.cylindricalWarp(img2, K)

				img,msk = self.stitchimg(img1,img2,mask)
				imgs.append(img)
				

			if (self.cylindrical == 1):
				h,w,c = self.left[-1].shape
				K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
				img = self.cylindricalWarp(self.right[-1],K)
				imgs.append(img)
			else:
				imgs.append(self.left[-1])


			if (len(imgs)>1):
				img,mask = self.stitchimg(imgs[0],imgs[1],None)
				for i in range(2,len(imgs),1):
					img,mask = self.stitchimg(imgs[i],img,mask)
				self.rimg = img
			else:
				if (self.cylindrical == 1):
					h,w,c = self.right[0].shape
					K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
					img = self.cylindricalWarp(self.right[0], K)
					self.rimg = img
				else:
					self.rimg = self.right[0]

			

	def mulstit1(self):
		if (self.center is None):
			#if center image is none, stitch left images and right images separately then stitch
			#them together.
			img,msk = self.stitchimg(self.limg,self.rimg,None)
			#save and show the finla panorama
			plt.imsave('stitched.png',img)
			plt.imshow(img),plt.show()

		else:
			#if center image is not none, stitch left images and the center image first then use the image to stitch
			#with the right image to form the final panorama
			center = None
			if (self.cylindrical == 1):
				f=500
				h,w,c = self.center.shape

				K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

				center = self.cylindricalWarp(self.center, K)
			else:
				center = self.center

			img,msk = self.stitchimg(self.limg,center,None)


			img,msk = self.stitchimg(self.rimg,img,msk)

			plt.imsave('stitched.png',img)
			plt.imshow(img),plt.show()
		



	def cylindricalWarp(self,img1, K):
		#get focal length from K[0,0]
		f = K[0,0]
		#get the width and height of the image
		im_h,im_w,__ = img1.shape
		
		cyl = np.zeros_like(img1)
		cyl_h,cyl_w,__ = cyl.shape
		#get center of the image
		x_c = float(cyl_w) / 2.0
		y_c = float(cyl_h) / 2.0
		for x_cyl in np.arange(0,cyl_w):
			for y_cyl in np.arange(0,cyl_h):
				#small angle approximation 
				theta = (x_cyl - x_c) / f
				#the height of the point in cylinder
				h     = (y_cyl - y_c) / f

				X = np.array([math.sin(theta), h, math.cos(theta)])
				X = np.dot(K,X)
				x_im = X[0] / X[2]

				#if the cylindrical warped point is out of the the image, then ignore
				if x_im < 0 or x_im >= im_w:
					continue

				y_im = X[1] / X[2]
				if y_im < 0 or y_im >= im_h:
					continue
				cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
		return (cyl)


	def stitchimg(self, img1, img2,msk):
		img22 = None
		img11 = None
		#for better accuracy, the image is in float type. But to find homography of two images,
		#the images should be in uint8. For the first two images with unit8 type, the assignment
		#could be done directely else transformed to uint8.
		if (isinstance(img2[0,0,0],float)):
			img22 = cv2.normalize(img2,None,0,255,cv2.NORM_MINMAX).astype('uint8')
		else:
			img22 = img2

		if (isinstance(img1[0,0,0],float)):
			img11 = cv2.normalize(img1,None,0,255,cv2.NORM_MINMAX).astype('uint8')
		else:
			img11 = img1

		#transform the rgb images into gray level images
		gray1 = cv2.cvtColor(img11,cv2.COLOR_RGB2GRAY)
		gray2 = cv2.cvtColor(img22,cv2.COLOR_RGB2GRAY)

		#use mask to get rid of surrounding noise in image
		if (msk is None):
			i=1
		else:
			img2[:,:,0] = img2[:,:,0] * msk
			img2[:,:,1] = img2[:,:,1] * msk
			img2[:,:,2] = img2[:,:,2] * msk

		#find homography of the images, source points and dest points.
		#use the homography to calculate the corner points for the two images
		#after transformation. Then use a matrix of proper size to fit in the
		#stitched image
		H, src_pts, dst_pts = self.matcher_obj.match(gray1,gray2)
		h,w,c = img1.shape
		corners = np.zeros((3,4))
		corners[0] = [0,0,w-1,w-1]
		corners[1] = [0,h-1,0,h-1]
		corners[2] = [1, 1, 1, 1]

		#corners array to fit in the corners of the images
		corners = H.dot(corners)
		corners[0] = corners[0]/corners[2]
		corners[1] = corners[1]/corners[2]
		corners[2] = corners[2]/corners[2]

		#find maximum row and column
		maxw = np.max(corners[0])
		maxh = np.max(corners[1])
		minw = np.min(corners[0])
		minh = np.min(corners[1])
		newh =0
		neww =0

		offset_y = 0
		offset_x = 0

		#offset to move the image is minw and minh
		offset_x = np.maximum(-minw,0)
		offset_y = np.maximum(-minh,0)
		dst_pts[:,:,0] = dst_pts[:,:,0] + offset_x
		dst_pts[:,:,1] = dst_pts[:,:,1] + offset_y

		#calculate the homography
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

		h1,w1,c1 = img2.shape

		#calculate the new width and height to fit in the new stitched image
		neww = np.maximum(maxw,w1) - np.minimum(0,minw)
		newh = np.maximum(maxh,h1) - np.minimum(0,minh)

		#warp the image for stitching
		img1_warp = cv2.warpPerspective(img1, M, (int(neww),int(newh)))


		#if img2 is float type, then newimg2 adding offset position is float
		#if it is uint8, then newimg2 is uint8 
		mask3 = np.zeros((int(newh),int(neww)))
		if (isinstance(img2[0,0,0],float)):
			newimg2 = np.zeros((int(newh),int(neww),3),dtype='float64')
		else:
			newimg2 = np.zeros((int(newh),int(neww),3),dtype='uint8')

		#newimg2 is the image adding width and height offsets for img2
		for i in range(0,h1):
			for j in range(0,w1):
				newimg2[i+int(offset_y),j+int(offset_x)] = img2[i,j]
				
		#add the two image for stitching and build mask for overlapping region
		img = (img1_warp + newimg2)
		
		#if the values of the img are greater than the values in newimg2, then the region
		#is regarded as overlapped area and set to 1 in mask3
		for i in range(0+int(offset_y),h1+int(offset_y)):
			for j in range(0+int(offset_x),w1+int(offset_x)):

				a = (img[i,j]==newimg2[i,j])
				if (a.any()==False):
					mask3[i,j] = 1

		#find positions of overlapping area and make dividing line for masks to blend
		#here to find the max and minimun column and divided by 2 as the dividing line
		#to make masks
		pos = np.where(mask3==1)
		minh = np.min(pos[0])
		maxh = np.max(pos[0])
		minw = np.min(pos[1])
		maxw = np.max(pos[1])

		#make mask for img1
		mask1 = np.zeros((int(newh),int(neww)))
		mask1 = img1_warp[:,:,0].copy()
		mask1[mask1>0] = 1

		#make mask for img2
		mask2 = np.zeros((int(newh),int(neww)))
		mask2 = newimg2[:,:,0].copy()
		mask2[mask2>0] = 1

		pos1 = np.where(mask1==1)
		max1 = np.max(pos1[1])
		min1 = np.min(pos1[1])


		pos2 = np.where(mask2 ==1)
		max2 = np.max(pos2[1])
		min2 = np.min(pos2[1])

		m = mask1 + mask2
		#to decide which image is in the left by comparing the minmum column of the image
		left = min1 < min2

		for i in range(int(newh)):
			for j in range(int(neww)):
				# if in overlapping area, for the left image, assign 1 for the mask of left part in
				#overlapping area and for the right image, assign 1 for the mask of right part in
				#overlapping area
				if (m[i,j]==2):

					if (j < int((maxw-minw)/2)+minw):
						if (left==1):
							mask1[i,j] = 1
							mask2[i,j] = 0
						else:
							mask2[i,j] = 1
							mask1[i,j] = 0

					if (j >= int((maxw-minw)/2)+minw):
						if (left==1):
							mask1[i,j] = 0
							mask2[i,j] = 1
						else:
							mask2[i,j] = 0
							mask1[i,j] = 1



		#imgblend for stitched and blending images		
		imgblend = np.zeros((img1_warp.shape[0],img1_warp.shape[1],img1_warp.shape[2]))

		img1_warp=img_as_float(img1_warp)
		newimg2=img_as_float(newimg2)

		plt.imshow(img1_warp),plt.show()
		#do image blending in the three channels
		imgblend[:,:,0] = self.blend.blend(img1_warp[:,:,0], newimg2[:,:,0], 10, mask1, mask2)
		imgblend[:,:,1] = self.blend.blend(img1_warp[:,:,1], newimg2[:,:,1], 10, mask1, mask2)
		imgblend[:,:,2] = self.blend.blend(img1_warp[:,:,2], newimg2[:,:,2], 10, mask1, mask2)
		msk = mask1+mask2 

		#use msk to get rid of noise parts around stitched image
		imgblend[:,:,0] = imgblend[:,:,0] * msk
		imgblend[:,:,1] = imgblend[:,:,1] * msk
		imgblend[:,:,2] = imgblend[:,:,2] * msk
		
		return imgblend,msk


if __name__ == '__main__':

	try:
		file = sys.argv[1]
		if os.path.exists(file):
			i=1
		else:
			file = "files1.txt"
	except:
		file = "files1.txt"

	#whether or not to use cylindrical warping
	try:
		ifcyl = sys.argv[2]
	except:
		ifcyl = 1

	#whether or not to use resizing
	try:
		ifresize = sys.argv[3]
	except:
		ifresize = 0

	#set resize height and width
	try:
		height = sys.argv[4]
		width = sys.argv[5]
	except:
		height = 500
		width = 400


	s = Stitch(file,int(ifcyl),int(ifresize),int(width),int(height))


	print('Start stitching...')
	s.lrList()
	s.leftStitch()
	s.rightStitch()
	s.mulstit1()
	


