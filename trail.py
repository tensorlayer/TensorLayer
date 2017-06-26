import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt


os.chdir("AudioFiles_BarkSequences")
sigbag = wavfile.read("vp_005_01_alone.wav")
sig = sigbag[1]
print ("The length of signal is: ", len(sig))
vec = signal.ricker(sig, 4.0)
plt.imshow(vec)
