import vamp
import librosa
data, rate = librosa.load("example.wav")
chroma = vamp.collect(data, rate, "nnls-chroma:nnls-chroma")
chroma
{'matrix': ( 0.092879819, array([[  61.0532608 ,   60.27478409,   59.3938446 , ...,  182.13394165,
          42.40084457,  116.55457306],
       [  68.8901825 ,   63.98115921,   60.77633667, ...,  245.88218689,
          68.51251984,  164.70120239],
       [  58.59794617,   50.3429184 ,   45.44804764, ...,  258.02362061,
          83.95749664,  179.91200256],
       ...,
       [   0.        ,    0.        ,    0.        , ...,    0.        ,
           0.        ,    0.        ],
       [   0.        ,    0.        ,    0.        , ...,    0.        ,
           0.        ,    0.        ],
       [   0.        ,    0.        ,    0.        , ...,    0.        ,
           0.        ,    0.        ]], dtype=float32))}
stepsize, chromadata = chroma["matrix"]
import matplotlib.pyplot as plt
plt.imshow(chromadata)
<matplotlib.image.AxesImage object at 0x7fe9e0043fd0>
plt.show()
