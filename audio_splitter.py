import librosa as lr
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

FREQUENCY_BINS = [0,86.47058824,172.94117647,259.41176471,
                  345.88235294,432.35294118,518.82352941,605.29411765,
                  691.76470588,778.23529412,864.70588235,951.17647059,
                  1037.64705882,1124.11764706,1210.58823529,1297.05882353,
                  1383.52941176,1470,1556.47058824,1642.94117647,
                  1729.41176471,1815.88235294,1902.35294118,1988.82352941,
                  2075.29411765,2161.76470588,2248.23529412,2334.70588235,
                  2421.17647059,2507.64705882,2594.11764706,2680.58823529,
                  2767.05882353,2853.52941176,2940,3026.47058824,
                  3112.94117647,3199.41176471,3285.88235294,3372.35294118,
                  3458.82352941,3545.29411765,3631.76470588,3718.23529412,
                  3804.70588235,3891.17647059,3977.64705882,4064.11764706,
                  4150.58823529,4237.05882353,4323.52941176,4410,
                  4496.47058824,4582.94117647,4669.41176471,4755.88235294,
                  4842.35294118,4928.82352941,5015.29411765,5101.76470588,
                  5188.23529412,5274.70588235,5361.17647059,5447.64705882,
                  5534.11764706,5620.58823529,5707.05882353,5793.52941176,
                  5880,5966.47058824,6052.94117647,6139.41176471,
                  6225.88235294,6312.35294118,6398.82352941,6485.29411765,
                  6571.76470588,6658.23529412,6744.70588235,6831.17647059,
                  6917.64705882,7004.11764706,7090.58823529,7177.05882353,
                  7263.52941176,7350,7436.47058824,7522.94117647,
                  7609.41176471,7695.88235294,7782.35294118,7868.82352941,
                  7955.29411765,8041.76470588,8128.23529412,8214.70588235,
                  8301.17647059,8387.64705882,8474.11764706,8560.58823529,
                  8647.05882353,8733.52941176,8820,8906.47058824,
                  8992.94117647,9079.41176471,9165.88235294,9252.35294118,
                  9338.82352941,9425.29411765,9511.76470588,9598.23529412,
                  9684.70588235,9771.17647059,9857.64705882,9944.11764706,
                  10030.58823529,10117.05882353,10203.52941176,10290.,
                  10376.47058824,10462.94117647,10549.41176471,10635.88235294,
                  10722.35294118,10808.82352941,10895.29411765,10981.76470588,
                  11068.23529412,11154.70588235,11241.17647059,11327.64705882,
                  11414.11764706,11500.58823529,11587.05882353,11673.52941176,
                  11760,11846.47058824,11932.94117647,12019.41176471,
                  12105.88235294,12192.35294118,12278.82352941,12365.29411765,
                  12451.76470588,12538.23529412,12624.70588235,12711.17647059,
                  12797.64705882,12884.11764706,12970.58823529,13057.05882353,
                  13143.52941176,13230,13316.47058824,13402.94117647,
                  13489.41176471,13575.88235294,13662.35294118,13748.82352941,
                  13835.29411765,13921.76470588,14008.23529412,14094.70588235,
                  14181.17647059,14267.64705882,14354.11764706,14440.58823529,
                  14527.05882353,14613.52941176,14700,14786.47058824,
                  14872.94117647,14959.41176471,15045.88235294,15132.35294118,
                  15218.82352941,15305.29411765,15391.76470588,15478.23529412,
                  15564.70588235,15651.17647059,15737.64705882,15824.11764706,
                  15910.58823529,15997.05882353,16083.52941176,16170.,
                  16256.47058824,16342.94117647,16429.41176471,16515.88235294,
                  16602.35294118,16688.82352941,16775.29411765,16861.76470588,
                  16948.23529412,17034.70588235,17121.17647059,17207.64705882,
                  17294.11764706,17380.58823529,17467.05882353,17553.52941176,
                  17640,17726.47058824,17812.94117647,17899.41176471,
                  17985,18072,18158,18245,
                  18331,18418,18504,18591,
                  18677,18764,18850,18937,
                  19023,19110,19196.47058824,19282.94117647,
                  19369,19455.88235294,19542.35294118,19628.82352941,
                  19715,19801.76470588,19888.23529412,19974.70588235,
                  20061,20147,20234.11764706,20320.58823529,
                  20407,20493,20580,20666.47058824,
                  20752,20839,20925.88235294,21012.35294118,
                  21098,21185,21271,21358.23529412,
                  21444,21531,21617,21704.11764706,
                  21790,21877,21963,22050]

freq_sorts = [sub_freq := [16,60],
    bass_freq := [60,250],
    lower_freq := [250,500],
    mid_freq := [500,2000],
    higher_freq := [2000,4000],
    presence_freq := [4000,6000],
    brilliance_freq := [6000,20000]]

def avg(array):
    return sum(array)/len(array)

def rev(array):
    return array[::-1]

def split(file, SIGMA):
    #plt.figure(figsize=(14, 5))
    frame, sr = lr.load(file)


    # -------------------------
    #get audio duration in seconds
    # -------------------------
    print("Getting audio duration")
    sample_duration = 1 / sr
    duration = sample_duration * len(frame)


    # -------------------------
    #get centroids for general color
    # -------------------------
    print("Getting Centroids")
    centr = lr.feature.spectral_centroid(frame, sr = sr)[0]
    centroids = []

    for c in centr:
        centroids.append(int(c))
    smooth_centroid = gaussian_filter1d(centroids,sigma = SIGMA)
    #print(len(smooth_centroid), len(frame), len(frame)/len(smooth_centroid))

    # -------------------------
    #get tempo
    # -------------------------
    print("getting BPM")
    onset_envelope = lr.onset.onset_strength(frame, sr= sr)
    tempo = lr.beat.tempo(onset_envelope=onset_envelope, sr = sr)
    tempo = int(tempo+0.5)


    # -------------------------
    #get amplitude of frequenzy of slices and spectra
    # -------------------------
    print("getting spectra and amplitudes")
    slices = [] #all slices
    single_slice = [] #buffer for a single slice

    slice_duration = SIGMA*4
    # slices the song
    for f in frame:
        if len(single_slice) == slice_duration:
            slices.append(single_slice)
            single_slice = []
        single_slice.append(f)

    spectra = [] #list of all spectras corresponding to slices

    #gets spectra and frequency range
    for i, slice in enumerate(slices):
        ft = np.fft.fft(slice) #fourier transform
        magnitude_spectrum = np.abs(ft)

        spectra.append(magnitude_spectrum)

        '''
        if i % (SIGMA*4/32) == 0 or i % (SIGMA*4/48) == 0:
            plt.plot(frequency, magnitude_spectrum)
            plt.xlabel('Frequenzy')
            plt.title("Time: {}".format(sample_duration*SIGMA*4*i))
            plt.draw()
            plt.pause(sample_duration)
            plt.clf()
        '''

    #gets frequency amplitudes
    freq_ampl = [
        sub_ampl := [],
        bass_ampl := [],
        lower_ampl := [],
        mid_ampl := [],
        higher_ampl := [],
        presence_ampl := [],
        brilliance_ampl := []
    ]
    rev_freq_ampl = rev(freq_ampl)
    rev_freq_sorts = rev(freq_sorts)

    for spectrum in spectra:
        freq_buffer = [
            sub_buffer  := [],
            bass_buffer  := [],
            lower_buffer := [],
            mid_buffer  := [],
            higher_buffer  := [],
            presence_buffer := [],
            brilliance_buffer := []
        ]
        rev_freq_buffer = rev(freq_buffer)
        
        for i in range(len(FREQUENCY_BINS)):
            for y in range(len(freq_sorts)):
                if FREQUENCY_BINS[i] - rev_freq_sorts[y][0] > 0 and FREQUENCY_BINS[i] <= brilliance_freq[1]:
                    rev_freq_buffer[y].append(spectrum[i])

        for y in range(len(freq_sorts)):
            rev_freq_ampl[y].append(avg(rev_freq_buffer[y]))

    freq_ampl = rev(rev_freq_ampl)

    print("Finnished splitting")
    return smooth_centroid,spectra,freq_ampl,tempo,duration,slice_duration









