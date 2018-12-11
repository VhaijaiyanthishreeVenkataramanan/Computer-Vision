# Google Colab Instructions

## 1. Open notebook
- go to [google colab](https://colab.research.google.com) and use this github url
	* https://github.com/Rice-Field/SIFT-Panorama-Stitch
- open the notebook
	* file -> save a copy in drive
	* or edit the file and download when done

## 2. Install libraries
Make a new cell at the top with:

	!pip install opencv-python==3.4.2.16
	!pip install opencv-contrib-python==3.4.2.16
	!pip install imutils

## 3. Retrieve images
replace image reading under introduction:

	image1 = cv2.imread("bryce_left_02.png")
	image2 = cv2.imread("bryce_right_02.png")

with:

	!wget "https://github.com/Rice-Field/SIFT-Panorama-Stitch/blob/master/images/bryce_left_02.png?raw=1"
	!wget "https://github.com/Rice-Field/SIFT-Panorama-Stitch/blob/master/images/bryce_right_02.png?raw=1"

	image1 = cv2.imread("bryce_left_02.png?raw=1")
	image2 = cv2.imread("bryce_right_02.png?raw=1")