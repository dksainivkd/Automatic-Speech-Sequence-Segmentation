import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from skimage.transform import radon, iradon
np.seterr(divide='ignore', invalid='ignore')
from scipy.fftpack import fft, dct

def get_mean(arr):
	s = 0.0
	for i in arr:
		s = s+ i
	return s/len(arr)
def get_std(arr):
	m = get_mean(arr)
	s = 0.0
	for i in arr:
		s = s+ (m-i)*(m-i)
	return s/len(arr)

def flatten(array):
	temp = []
	for sub in array:
		for item in sub:
			temp.append(item)	
	return temp

def gen_spectogram_image(spec):
	rows = len(spec)
	cols = len(spec[0])
	img = np.zeros((rows,cols),dtype= np.uint8)

	'''
	assuming the values in the trasform varies from 10^-14 to 10^2
	the corresponding non-linear mapping gave the equation
	y = 16 log10(x) + 224
	'''
	for i in range(0,rows):
		for j in range(0,cols):
			try:
				img[i][j] = int(math.log10(spec[i][j]) * 16 + 224)
			except ValueError:
				img[i][j] = 0

	return img

def get_log_image(spec):
	rows = len(spec)
	cols = len(spec[0])
	img = np.zeros((rows,cols),dtype= np.float32)

	'''
	assuming the values in the trasform varies from 10^-14 to 10^2
	the corresponding non-linear mapping gave the equation
	y = 16 log10(x) + 224
	'''
	for i in range(0,rows):
		for j in range(0,cols):
			try:
				img[i][j] = math.log10(spec[i][j])
			except ValueError:
				img[i][j] = 0

	return img
	
def gen_energy_histogram(ener):
	'''normally log of energy for a frame is not more than 3 and for many
	silence parts it is as low as -8 so bins from bins have been taken 
	[-12 , -12 + 0.17d], here d = {0,1,2,...,99}'''

	y = np.linspace(-12 , 5 , num=100)
	plt.hist(ener , y , alpha = 0.5)
	plt.show()

def estimate_threshold_VAD(ener):
	'''it was observed that the log energies of the frame approximates somewhat
	a mixture of two gaussians or two clusters, and the same was used to estimate 
	the threshold
	'''
	N = 2

	CV = 'full'
	gmm =GMM(n_components = N , covariance_type = CV)
	gmm.fit(ener)


	kmeans = KMeans(n_clusters = N , random_state = 0).fit(ener)
	return gmm , kmeans

def get_radon_projections(spectogram):
	projections1 = flatten(radon(spectogram, theta=[22.5]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections2 = flatten(radon(spectogram, theta=[45]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections3 = flatten(radon(spectogram, theta=[67.5]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections4 = flatten(radon(spectogram, theta=[90]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections5 = flatten(radon(spectogram, theta=[112.5]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections6 = flatten(radon(spectogram, theta=[135]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections7 = flatten(radon(spectogram, theta=[157.5]))
	#RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]

	return np.array([projections1 , projections2 , projections3, projections4, projections5,
		projections6, projections7])

def get_correlation(spec1 , spec2, dc=False):

	if(len(spec1) > len(spec2)):
		vec1 = np.array(spec1)
		vec2 = np.array(spec2)
	else:
		vec1 = np.array(spec2)
		vec2 = np.array(spec1)
	
	m = len(vec1)
	k = len(vec2)
	if((m-k)/k < 0.5):
		radon_pr1 = get_radon_projections(vec1[:k])
		radon_pr2 = get_radon_projections(vec2)
		corr = np.corrcoef(radon_pr1 , radon_pr2)
		return (corr.mean(), corr.std())
	else:
		skip = int(k * k)
		pstart = 0
		pstop = k
		corr = []
		while pstop < m :

			radon_pr1 = get_radon_projections(vec1[pstart:pstop])
			radon_pr2 = get_radon_projections(vec2[pstart:pstop])
			for i in range(0,len(radon_pr1)):
				if(get_std(radon_pr1[i]) == 0 or get_std(radon_pr2[i])==0):
					continue
				if(not(dc)):
					v_temp = np.corrcoef(radon_pr1[i],radon_pr2[i])
				else:
					v_temp = np.corrcoef(dct(radon_pr1[i]),dct(radon_pr2[i]))
				corr.append(v_temp[0][1])
			pstart = pstart + skip
			pstop = pstop + skip

		return np.array((get_mean(corr),get_std(corr)))

	'''
	radon_pr1 = get_radon_projections(vec1)
	radon_pr2 = get_radon_projections(vec2)

	corr =  np.corrcoef(radon_pr1,radon_pr2)

	return corr.mean()
	'''


def set_labels(mat):
	rows = len(mat)
	label = np.zeros((rows,rows))
	for i in range(0,rows):
		for j in range(0,rows):
			if(mat[i][j] > -0.05):
				label[i][j] = i
			else:
				label[i][j] = -1
	return label
