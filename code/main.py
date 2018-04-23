import essentia
from essentia.standard import *
import cv2
import sys
import os
import numpy as np
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
from functions import *
import math
from skimage.transform import radon, iradon

#plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

fn = sys.argv[1]		#audio file name
sr = int(sys.argv[2])		#sampling rate of the audio

loader=essentia.standard.MonoLoader(filename =fn, sampleRate = sr )
audio = loader()		#audio sequence

#plot(audio[1*41100:2*41100])
#plt.title("input audio signal")
#show()

#resample the audio to 16kHz
if(sr>20000):
	rs = Resample(inputSampleRate = sr, outputSampleRate = 16000 , quality = 0)
	audio = rs(audio)
	sr = 16000

n_samples = len(audio)		#number of samples in the audio

#print n_samples

w =  Windowing(type = 'hann')
wl = sr/1000*25			#window length considering a window of 25ms
overlap = sr/1000*15		#overlap between two consecutive windows 10ms


spectrum = Spectrum(size=wl)
energy = Energy()
ener=[]

spectogram = []
fstart = 0
fstop = fstart+wl
while fstop < n_samples:
	frame = audio[fstart:fstop]
        #print energy(frame)
        if (energy(frame)>1e-10):
                e = math.log10(energy(frame))
                ener.append(e)		#log of total energy is being taken

	fstart = fstart + overlap
	fstop = fstop + overlap



#hist = gen_energy_histogram(ener)		#draw and plot the histogram of energies
#plt.subplot(10 ,1, 1)
#plt.imshow(hist)
#show()


temp = np.array(ener)
temp = temp.reshape(-1,1)		#clustering demands the matrix in certain fashion
threshold_gmm , threshold_cluster = estimate_threshold_VAD(temp)	#threshold contains various GMM parametes

#use predict_proba(x) to evaluate posteriori

#we will use the GMM to evaluate the VAD
#this variable sores mean[0] < mean[1] of GMM 
m = threshold_gmm.means_[0] < threshold_gmm.means_[1]

spectogram = []			#this contains spectogram of all the frames
spectogram1 = []		#this contaisn spectogram of only speech portions
features = []			#this contains the spectrum of voiced portion of speech in 
				#in sliced fashion
spectogram_temp = []
wlf_silent = True		#stores whether or not last frame was silent

trans_sample = []		#stores the transitioning samples speech->silence and vice-versa

fstart = 0
fstop = fstart+wl
while fstop < n_samples:
	frame = audio[fstart:fstop]
	if (energy(frame)>1e-10):
		ener = [math.log10(energy(frame))]	

	#ener = [math.log10(energy(frame))]		#log of total energy is being taken
	spec = spectrum(w(frame))

	a = threshold_cluster.predict([ener])
	b = threshold_gmm.predict_proba([ener])
	
	spectogram.append(spec)

	#write your code here to separate the two spectograms
	if( (b[0][0] > b[0][1] and not(m)) or  (b[0][0] < b[0][1] and m)):
		spectogram1.append(spec)

		if(not(wlf_silent)):
			spectogram_temp.append(spec)
		wlf_silent = False

	else :
		wlf_silent = True
		if(len(spectogram_temp) > 0):

			trans_sample.append(fstart-overlap)

			features.append(spectogram_temp)
		spectogram_temp = []


	fstart = fstart + overlap
	fstop = fstop + overlap

'''
img1 = gen_spectogram_image(spectogram)
img2 = gen_spectogram_image(spectogram1)
plt.subplot(2,1, 1)
plt.imshow(np.rot90(img1))
plt.title("Spectrum of all frame ")
plt.xlabel('time (s)')
plt.ylabel('Amplitude')


plt.subplot(2,1, 2)
plt.imshow(np.rot90(img2))
plt.title("spectrum of only speech portion")
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()
exit()'''



#merging the frames
i = 0
while ( i < len(features)-1):

	diff = trans_sample[i+1]-trans_sample[i]
	len_frame_ip1 = len(features[i+1])
	n_sample_ip1 = 15*len_frame_ip1 + 10
	if(diff - n_sample_ip1 < 24000):
		features[i] = features[i] + features[i+1]
		features.pop(i+1)
		trans_sample.pop(i)
	else:
		i = i+1


'''img1 = gen_spectogram_image(features[0])
img2 = gen_spectogram_image(features[1])
img3 = gen_spectogram_image(features[2])
plt.subplot(2,1, 1)
plt.imshow(np.rot90(img1))
#plt.title("Spectrum of all frame ")
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()

plt.subplot(2,1, 2)
plt.imshow(np.rot90(img2))
#plt.title("spectrum of only speech portion")
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()

plt.subplot(2,1, 1)
plt.imshow(np.rot90(img3))
#plt.title("Spectrum of all frame ")
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()
exit()

'''
'''
for i in range(0, len(features)):
	cv2.imshow("adf",gen_spectogram_image(features[i]))
	cv2.waitkey(0)
'''


#print np.array(trans_sample)/16000.0,len(features)

log_features = []
for i in features:
	log_features.append(get_log_image(i))

corr_mat_mean = np.zeros((len(features),len(features)))
corr_mat_std = np.zeros((len(features),len(features)))
for i in range(0,len(log_features)):
	for j in range(i , len(log_features)):
		if(i == j ):
			corr_mat_mean[i][j] = 0.0
			corr_mat_std[i][j] = 0.0
			continue
		m = get_correlation(log_features[i],log_features[j],dc=True)		#return [mean , std ] of correlations
		corr_mat_mean[i][j] = math.log(m[0])
		corr_mat_mean[j][i] =math.log(m[0])
		corr_mat_std[i][j] = math.log(m[1])
		corr_mat_std[j][i] = math.log(m[1])

#print corr_mat_mean
plt.figure(figsize=(5,4))
for i in range(0,len(features)):
	print i
	plt.subplot(len(features),1,i+1)
	plt.plot(corr_mat_mean[i])
	
plt.show()



first = True

corarray = [[],[],[],[],[],[],[]]
for i in range (0, len(features)):
	img = np.array(features[i])

	projections1 = radon(img, theta=[22.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections2 = radon(img, theta=[45]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections3 = radon(img, theta=[67.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections4 = radon(img, theta=[90]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections5 = radon(img, theta=[112.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections6 = radon(img, theta=[135]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	projections7 = radon(img, theta=[157.5]) #RADON Projections for each non silent part || theta=[22.5, 45, 67.5, 90, 112.5, 135, 157.5]
	
	
	projections1 = flatten(projections1)
	projections2 = flatten(projections2)
	projections3 = flatten(projections3)
	projections4 = flatten(projections4)
	projections5 = flatten(projections5)
	projections6 = flatten(projections6)
	projections7 = flatten(projections7)

	projections1=np.array(projections1)
	projections2=np.array(projections2)
	projections3=np.array(projections3)
	projections4=np.array(projections4)
	projections5=np.array(projections5)
	projections6=np.array(projections6)
	projections7=np.array(projections7)
	
	'''print"shape of projections"
	print projections1.shape
	print projections2.shape
	print projections3.shape
	print projections4.shape
	print projections5.shape
	print projections6.shape
	print projections7.shape
	print "***********************"'''
	
	if(i>0):
		projections1=projections1[0:285]
		projections2=projections2[0:285]
		projections3=projections3[0:285]
		projections4=projections4[0:285]
		projections5=projections5[0:285]
		projections6=projections6[0:285]
		projections7=projections7[0:285]
		print a[0].shape, " i=1 ", projections1.shape
	



	#print "Current sample window: ", trans_sample[i]
	#print"shape of projection "
	#print projections1.shape
	#plt.plot(projections1)
	#plt.show()
	

	if(first is False):
		a=np.array(a);
		#print a[0],projections1
		corr1 = np.corrcoef(a[0].T,projections1.T)
		corr2 = np.corrcoef(a[1].T,projections2.T)
		corr3 = np.corrcoef(a[2].T,projections3.T)
		corr4 = np.corrcoef(a[3].T,projections4.T)
		corr5 = np.corrcoef(a[4].T,projections5.T)
		corr6 = np.corrcoef(a[5].T,projections6.T)
		corr7 = np.corrcoef(a[6].T,projections7.T)

		#print corr1
		corarray[0].append( (corr1[0][1]))
		corarray[1].append( (corr2[0][1]))
		corarray[2].append( (corr3[0][1]))
		corarray[3].append( (corr4[0][1]))
		corarray[4].append( (corr5[0][1]))
		corarray[5].append( (corr6[0][1]))
		corarray[6].append( (corr7[0][1]))
	a = np.stack((projections1,projections2,projections3,projections4,projections5,projections6,projections7))


	#show()
	
	first = False

corarray=np.array(corarray)
corarray=corarray.T
print corarray

spectogram = np.array(spectogram)
img = gen_spectogram_image(spectogram)
img1 = gen_spectogram_image(spectogram1)

height, width = img.shape[:2]
img = np.rot90(img)


plt.subplot(2 ,1, 2)
plt.imshow(img)
plt.title('COmpare')
plt.ylabel('Spectrogram')
plt.xlabel('Pixel')
plt.show()

plt.subplot(2,1, 2)
plt.imshow(np.rot90(img1))
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()

plt.subplot(4,1, 3)
plt.plot(audio[:])
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.show()



plt.subplot(10 , 1, 1)
plt.plot(corarray[0])
plt.xlabel('window')
plt.ylabel('correlation1')

plt.subplot(10 , 1, 2)
plt.plot(corarray[1])
plt.xlabel('window')
plt.ylabel('correlation2')



plt.subplot(10, 1, 3)
plt.plot(corarray[2])
plt.xlabel('window')
plt.ylabel('correlation3')


plt.subplot(10, 1, 4)
plt.plot(corarray[3])
plt.xlabel('window')
plt.ylabel('correlation4')


plt.subplot(10, 1, 5)
plt.plot(corarray[4])
plt.xlabel('window')
plt.ylabel('correlation5')


plt.subplot(10 , 1, 6)
plt.plot(corarray[5])
plt.xlabel('window')
plt.ylabel('correlation6')


plt.subplot(10, 1, 7)
plt.plot(corarray[6])
plt.xlabel('window')
plt.ylabel('correlation7')
plt.show()

#cv2.imshow("all",img)
#cv2.imshow("speech",img1)
#cv2.waitKey(0)


#cv2.destroyAllWindows()
