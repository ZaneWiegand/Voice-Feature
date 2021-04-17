# %%
import numpy as np
import sklearn
import librosa
import librosa.display
import matplotlib.pyplot as plt
# %%
x, sr = librosa.load('./train_sample/aloe/24EJ22XBZ5.wav')
# 绘制声波信号
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# 放大
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
# %%
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))
# %%
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)
# (2647,)
# 计算时间变量
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# 归一化频谱质心


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# 沿波形绘制频谱质心
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
# %%
spectral_rolloff = librosa.feature.spectral_rolloff(
    x, sr=sr, roll_percent=0.85)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='b')
# %%
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=512)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time',
                         y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
# %%
y, sr = librosa.load('./train_sample/aloe/24EJ22XBZ5.wav')
S = librosa.stft(y, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect')
S = np.abs(S)
D = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(20, 10))
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram of burger')
# %%
mel = librosa.feature.melspectrogram(x, sr=sr)
D = librosa.amplitude_to_db(np.abs(mel), ref=np.max)
plt.figure(figsize=(20, 10))
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram of burger')
# %%
