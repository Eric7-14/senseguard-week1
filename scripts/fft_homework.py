import numpy as np
import matplotlib.pyplot as plt

fs = 1000       # 采样率（Hz）
duration = 1.0  # 信号时长（秒）
# ⚠️ 你的学号后两位是 23，所以 20 + 23 = 43
f0 = 20 + 23    # 信号频率 (Hz)
noise = 0.8     # 噪声大小

# 1. 生成时间轴 t
t = np.arange(0, duration, 1 / fs)

# 2. 生成信号 x：正弦波 + 噪声
x = np.sin(2 * np.pi * f0 * t) + noise * np.random.default_rng().normal(size=t.size)

# 3. 算频谱
spectrum = np.abs(np.fft.rfft(x))

# 4. 算每个点对应的频率
freqs = np.fft.rfftfreq(t.size, 1 / fs)

# 5. 用 np.argmax 找频谱最高点的下标，打印这个下标对应的频率
max_index = np.argmax(spectrum)
max_freq = freqs[max_index]
print(f"找到的峰值频率是: {max_freq:.2f} Hz (目标 f0 是 {f0} Hz)")

# 6. 画上下两张图
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# 上图：前 0.2 秒的波形 (0.2 * 1000 = 200 个点)
axes[0].plot(t[:200], x[:200])
axes[0].set_title("Waveform (First 0.2s)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

# 下图：频谱
axes[1].plot(freqs, spectrum)
axes[1].set_title("Frequency Spectrum (FFT)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")

# 调整排版并保存到 output 文件夹
plt.tight_layout()
plt.savefig("output/fft.png")
