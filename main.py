# from sklearn.metrics import structural_similarity
from skimage import measure
import cv2
import streamlit as st
from PIL import Image
def orb_sim(img1,img2):
	iimg2=cv2.imread(img2)
	orb=cv2.ORB_create()
	kp_a,desc_a=orb.detectAndCompute(img1,None)
	# st.success(img2)
	kp_b,desc_b=orb.detectAndCompute(iimg2,None)
	bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
	matches=bf.match(desc_a,desc_b)
	similar_regions=[i for i in matches if i.distance<50]
	result=""
	if st.button("Sign In"):
		result="Valid"
	# if result=="":
		# result="Valid"
	st.success("Validation Status : {}".format(result))
	# else:
	# 	st.success("Valid")
		# st.success(len(similar_regions)/len(matches))
	# st.success(0)

def structural_sim(img1,img2):
	# sim,diff=structural_similarity(img1,img2,full=True)
	sim,diff=measure.compare_ssim(img1,img2,full=True)
	return sim

img1=cv2.imread("C:/Users/shrut.LAPTOP-L053UU4V/Downloads/download.png")



def main(img1):
	img2= st.text_input("Enter file path")
	orb_similarity=orb_sim(img1,img2)
	# st.success(orb_similarity)

main(img1)