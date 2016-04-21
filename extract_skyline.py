# USAGE
# python extract_skyline.py

# Import the necessary packages
from __future__ import print_function
import numpy as np
import mahotas
import cv2


count = 1


def canny_extract(image_count):
	# ------------------------------------------------------------------------------------------------
	# START EXTRACTING SKYLINE FROM IMAGES
	# ------------------------------------------------------------------------------------------------

	# READ IN IMAGES FROM FILE
	image = cv2.imread("images/" + str(image_count) + ".jpg")

	# MAKE COPY OF ORIGINAL IMAGE
	original = image.copy()

	# DISPLAY ORIGINAL IMAGE, CONVERT TO GRAYSCALE AND BLUR THE IMAGE
	cv2.imshow("STEP 1: Original", image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.GaussianBlur(image, (5,5), 0)
	#cv2.imwrite("images/blurred/blurred" + str(image_count) + ".jpg", image)
	#cv2.imshow("STEP 2: Blurred", image)

	# APPLY CANNY EDGE DETECTION TO THE BLURRED IMAGE
	canny = cv2.Canny(image, 30, 120)
	#cv2.imwrite("images/canny/canny" + str(image_count) + ".jpg", canny)
	#cv2.imshow("STEP 3: canny edge detection", canny)

	# BLUR THE CANNY DETECTION IMAGE TO MERGE GAPS LEFT IN BETWEEN THE LINES
	blurCanny = cv2.GaussianBlur(canny, (7,7), 0)
	#cv2.imwrite("images/blur_canny/blur_canny" + str(image_count) + ".jpg", blurCanny)

	#cv2.imshow("STEP 4: blurred canny edge detection", blurCanny)

	# FLOODFILL THE TOP OF THE IMAGE TO WHITE
	diff = (6,6,6)
	h, w = blurCanny.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(blurCanny,mask,(0,0),(255,255,255),diff,diff)
	#cv2.imshow("STEP 5: floodFillTopWhite", blurCanny)
	#cv2.imwrite("images/floodfill/floodFillTopWhite" + str(image_count) + ".jpg", blurCanny)

	#print("h: " + str(h) + " w: " + str(w))

	# SIMPLE THRESHOLD THE BLURRED CANNY IMAGE
	(T, blurCannyThresh) = cv2.threshold(blurCanny, 155, 255, cv2.THRESH_BINARY)
	#cv2.imshow("STEP 6: Threshold Binary Inverse", blurCannyThresh)
	#cv2.imwrite("images/thresholding/ThresholdBinaryInverse" + str(image_count) + ".jpg", blurCannyThresh)

	# FLOODFILL THE BOTTOM OF THE IMAGE TO GREY
	diff = (6,6,6)
	h, w = blurCannyThresh.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(blurCannyThresh,mask,(0,h-1),(145,255,0),diff,diff)
	#cv2.imshow("STEP 7: floodFillBottomGrey", blurCannyThresh)
	#cv2.imwrite("images/floodfill/floodFillBottomGrey" + str(image_count) + ".jpg", blurCannyThresh)

	# FLOODFILL THE TOP OF THE IMAGE TO BLACK
	diff = (6,6,6)
	h, w = blurCannyThresh.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(blurCannyThresh,mask,(0,0),(0,0,0),diff,diff)
	#cv2.imshow("STEP 8: floodFillTopBlack", blurCannyThresh)
	#cv2.imwrite("images/floodfill/floodFillTopBlack" + str(image_count) + ".jpg", blurCannyThresh)

	# FloodFill the tom of the image to white again
	diff = (6,6,6)
	h, w = blurCannyThresh.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(blurCannyThresh,mask,(0,0),(255,255,255),diff,diff)
	#cv2.imshow("STEP 9: floodFillTopWhiteAgain", blurCannyThresh)
	#cv2.imwrite("images/floodfill/floodFillTopWhiteAgain" + str(image_count) + ".jpg", blurCannyThresh)

	# FLOODFILL THE BOTTOM OF THE IMAGE TO GREY
	diff = (6,6,6)
	h, w = blurCannyThresh.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(blurCannyThresh,mask,(0,h-1),(0,0,0),diff,diff)
	#cv2.imshow("STEP 10: floodFillBottomBlack", blurCannyThresh)
	#cv2.imwrite("images/floodfill/floodFillBottomBlack" + str(image_count) + ".jpg", blurCannyThresh)


	# MASK THE IMAGE TO DISPLAY ONLY THE SKYLINE
	masked = cv2.bitwise_and(original, original, mask = blurCannyThresh)
	#cv2.imshow("STEP 11: Extracted Sky Region", masked)
	cv2.imwrite("images/sky_region/ExtractedSky" + str(image_count) + ".jpg", masked)

# ------------------------------------------------------------------------------------------------
# FUNCTION TO FIND DARK CLOUDS FORM OVERCAST SKY
# ------------------------------------------------------------------------------------------------
def dark_clouds(image_count):

	# LOAD THE IMAGE, CONVERT IT TO GRAYSCALE, AND BLUR IT
	# SLIGHTLY TO REMOVE HIGH FREQUENCY EDGES THAT WE AREN'T
	# INTERESTED IN
	img = cv2.imread("images/sky_region/ExtractedSky" + str(image_count) + ".jpg")
	original = img.copy()

	# FIND AND COUNT THE NUMBER OF BLACK PIXELS IN AN IMAGE
	BLACK = np.array([0,0,0],np.uint8)
	blackRange = cv2.inRange(img,BLACK,BLACK)
	no_black_pixels = cv2.countNonZero(blackRange)

	# CONVERT BGR TO HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# DEFINE RANGE OF GREY COLOR IN HSV
	lower_grey = np.array([90,0,0], dtype=np.uint8)
	upper_grey = np.array([130,255,125], dtype=np.uint8)

	# THRESHOLD THE HSV IMAGE TO GET ONLY GREY COLORS
	mask = cv2.inRange(hsv, lower_grey, upper_grey)

	# COUNT NUMBER OF GREY PIXELS
	no_grey_pixels = cv2.countNonZero(mask)

	# BITWISE-AND MASK AND ORIGINAL IMAGE
	res = cv2.bitwise_and(original,original, mask= mask)

	'''cv2.imshow("masked grey sky",res)'''

	# GET THE TOTAL NUMBER OF PIXELS
	total_pixels = original.size / 3

	# GET THE NUMBER OF PIXELS IN THE SKY REGION OF AN IMAGE
	sky_region_pixels = total_pixels - no_black_pixels


	# CALCULATE THE PERCENTAGE OF THE COLOUR grey PRESENT IN THE SKY
	if no_grey_pixels == 0:
		return("There is no grey pixels in the image." )
	else:
		grey_percentage = (no_grey_pixels / sky_region_pixels) * 100

		'''print("The total number of pixels is: " + str(total_pixels))
		print("The number of grey pixels is: " + str(no_grey_pixels))
		print("The number of black pixels is: " + str(no_black_pixels))
		print("The number of pixels of the sky region is : " + str(sky_region_pixels))'''
		print("The percentage of grey in the sky region is : " + str(grey_percentage))

		if grey_percentage > 70:
			return("Severe stormy skies.\n" )
		elif grey_percentage > 50 and grey_percentage <= 70:
			return("Very Stormy skies.\n" )
		elif grey_percentage > 30 and grey_percentage <= 50:
			return("Some stormy skies.\n" )
		elif grey_percentage > 9 and grey_percentage <= 30:
			return("Scattered rain clouds.\n" )
		else:
			return("The sky is overcast.\n")

def thresholding_extract(image_count):
	# READ IN IMAGES FROM FILE
	image = cv2.imread("images/" + str(image_count) + ".jpg")
	original = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(image, (5, 5), 0)

	# FINDING THE THRESHOLD VALUE 'T'  AND APPLYING IT TO THE IMAGE
	T = mahotas.thresholding.otsu(blurred)

	thresh = image.copy()
	thresh[thresh > T] = 255
	thresh[thresh < 255] = 0
	thresh = cv2.bitwise_not(thresh)

	(T, threshInv) = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV)
	#cv2.imwrite("images/otsu/otsu" + str(image_count) + ".jpg", threshInv)


	# FLOODFILL THE BOTTOM OF THE IMAGE TO GREY
	diff = (6,6,6)
	h, w = threshInv.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(threshInv,mask,(0,h-1),(145,255,0),diff,diff)
	#cv2.imshow("STEP 7: floodFillBottomGrey", threshInv)
	#cv2.imwrite("images/floodfill1/floodFillBottomGrey" + str(image_count) + ".jpg", threshInv)

	# FLOODFILL THE TOP OF THE IMAGE TO BLACK
	diff = (6,6,6)
	h, w = threshInv.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(threshInv,mask,(0,0),(0,0,0),diff,diff)
	#cv2.imshow("STEP 8: floodFillTopBlack", threshInv)
	#cv2.imwrite("images/floodfill1/floodFillTopBlack" + str(image_count) + ".jpg", threshInv)

	# FLOODFILL THE BOTTOM OF THE IMAGE TO GREY
	diff = (6,6,6)
	h, w = threshInv.shape[:2]
	mask = np.zeros((h+2, w+2), dtype = "uint8")
	mask[:] = 0
	cv2.floodFill(threshInv,mask,(0,h-1),(255,255,255),diff,diff)
	#cv2.imshow("STEP 7: floodFillBottomWhite", threshInv)
	#cv2.imwrite("images/floodfill1/floodFillBottomWhite" + str(image_count) + ".jpg", threshInv)

	(T, thresh) = cv2.threshold(threshInv, 155, 255, cv2.THRESH_BINARY_INV)
	#cv2.imwrite("images/inverse/inverse" + str(image_count) + ".jpg", thresh)
	#cv2.imshow("Threshold Binary", thresh)

	masked = cv2.bitwise_and(original, original, mask = thresh)
	#cv2.imshow("STEP 11: Extracted Sky Region", masked)
	cv2.imwrite("images/sky_region/ExtractedSky" + str(image_count) + ".jpg", masked)

# ------------------------------------------------------------------------------------------------
# START COUNTING PIXEL COLOURS
# ------------------------------------------------------------------------------------------------
def count_pixels(image_count):

	image = cv2.imread("images/sky_region/ExtractedSky" + str(image_count) + ".jpg")
	#cv2.imshow("Extracted Sky Region", image)

	original = image.copy()

	# FIND AND COUNT THE NUMBER OF BLACK PIXELS IN AN IMAGE
	BLACK = np.array([0,0,0],np.uint8)
	blackRange = cv2.inRange(image,BLACK,BLACK)
	no_black_pixels = cv2.countNonZero(blackRange)

	# Convert BGR to HSV
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# DEFINE RANGE OF BLUE COLOR IN HSV
	lower_blue = np.array([0,90,0], dtype=np.uint8)
	upper_blue = np.array([130,255,255], dtype=np.uint8)

	# DEFINE RANGE OF BLUE COLOR IN HSV
	lower_white = np.array([0,40,220], dtype=np.uint8)
	upper_white = np.array([109,89,255], dtype=np.uint8) 

	# THRESHOLD THE HSV IMAGE TO GET ONLY BLUE COLORS
	blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

	white_mask = cv2.inRange(hsv, lower_white, upper_white)

	total_mask = blue_mask - white_mask

	# COUNT NUMBER OF BLUE PIXELS
	no_blue_pixels = cv2.countNonZero(total_mask)

	# BITWISE-AND MASK AND ORIGINAL IMAGE
	res = cv2.bitwise_and(original,original, mask= total_mask)

	#cv2.imshow("masked blue sky",res)
	cv2.imwrite("images/masked_sky/maskedSky" + str(image_count) + ".jpg", res)

	# GET THE TOTAL NUMBER OF PIXELS
	total_pixels = original.size / 3

	# GET THE NUMBER OF PIXELS IN THE SKY REGION OF AN IMAGE
	sky_region_pixels = total_pixels - no_black_pixels

	if sky_region_pixels < 5000:
		thresholding_extract(image_count)
		count_pixels(image_count)

	# CALCULATE THE PERCENTAGE OF THE COLOUR BLUE PRESENT IN THE SKY
	elif no_blue_pixels == 0:
		print("There is no blue pixels in the image." )
	else:
		blue_percentage = (no_blue_pixels / sky_region_pixels) * 100

		print("The total number of pixels is: " + str(total_pixels))
		print("The number of blue pixels is: " + str(no_blue_pixels))
		print("The number of black pixels is: " + str(no_black_pixels))
		print("The number of pixels of the sky region is : " + str(sky_region_pixels))
		print("The percentage of blue in the sky region is : " + str(blue_percentage))

		if blue_percentage > 90:
			print("Clear blue sky.\n" )
		elif blue_percentage > 70 and blue_percentage <= 90:
			print("Clear sky with some clouds.\n" )
		elif blue_percentage > 50 and blue_percentage <= 70:
			print("Clear and cloudy sky.\n" )
		elif blue_percentage > 30 and blue_percentage <= 50:
			print("Cloudy sky with some clear blue sky.\n" )
		elif blue_percentage > 10 and blue_percentage <= 30:
			print("Cloudy sky.\n" )
		else:
			print (dark_clouds(image_count))

	# RETURN THE NUMBER OF PIXELS IN THE SKY REGION
	return (sky_region_pixels)


def read_image():
	global count

	image = cv2.imread("images/" + str(count) + ".jpg")

	print("\n---------------------------------------------")
	print (count)

	# CALL FUNCTION TO EXTRACT SKY REGION FROM AN IMAGE
	canny_extract(count) 

	count_pixels(count)

	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
	    cv2.destroyAllWindows()
	elif k == 32: # wait for 'space' key to save and exit
		count = count + 1
		read_image()

if __name__ == "__main__":
	read_image()
