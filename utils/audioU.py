import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile


def pre_process(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    samples = np.array(audio.get_array_of_samples())

    # 获取音频文件的采样率
    rate, data = wavfile.read(audio_file)

    # 使用noisereduce库进行降噪
    reduced_noise = nr.reduce_noise(y=samples, sr=rate)

    # 将降噪后的数据转换回AudioSegment
    reduced_noise_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    # # 降低采样率到16000 Hz
    # reduced_noise_audio = reduced_noise_audio.set_frame_rate(16000)

    output_dir = "./processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存降噪和降采样后的音频文件
    filename = os.path.basename(audio_file)
    output_file = os.path.join(output_dir, filename)
    reduced_noise_audio.export(output_file, format="wav")

    return output_file
