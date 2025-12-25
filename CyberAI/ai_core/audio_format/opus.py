import os
import subprocess
import wave
from pydub import AudioSegment
import opuslib_next
import numpy as np
import pickle  
from typing import List 
from utils.util import Util  # 导入 Util 类

class Opus_Encoder:
    _instance = None

    def __new__(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Opus_Encoder, cls).__new__(cls)
        return cls._instance

    def __init__(self, ffmpeg_path=None):
        self.sample_rate = 16000
        self.channel = 1
        self.sample_width = 2
        self.opus_sample_rate = 16000
        self.opus_channel = 1
        self.opus_sample_width = 2
        self.opus_frame_time = 60
        self.opus_frame_size = int(self.opus_sample_rate * self.opus_frame_time / 1000)
        
        # 如果没有提供 ffmpeg 路径，则使用相对路径
        if ffmpeg_path is None:
            # 使用 Util.get_process_dir() 获取项目根目录，然后拼接 FFmpeg 路径
            self.ffmpeg_path = os.path.join(Util.get_process_dir(), 'ai_core', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe')
        else:
            # 如果提供的是相对路径，转换为绝对路径
            self.ffmpeg_path = os.path.abspath(ffmpeg_path)
        
        # 设置环境变量，确保能找到 FFmpeg
        os.environ['PATH'] = os.path.dirname(self.ffmpeg_path) + os.pathsep + os.environ['PATH']

    def convert_to_pcm_with_ffmpeg(self, audio_file_path):
        """使用 FFmpeg 将音频文件转换为 PCM 格式"""
        # 验证输入文件是否存在
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")
        
        # 使用输入文件的目录创建临时文件，而不是固定路径
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        temp_dir = os.path.dirname(audio_file_path) or '.'
        temp_pcm_path = os.path.join(temp_dir, f"{base_name}_temp.pcm")
        
        # 使用 FFmpeg 转换为 PCM
        cmd = [
            self.ffmpeg_path,
            '-i', audio_file_path,  # 输入文件
            '-ar', str(self.sample_rate),  # 采样率
            '-ac', str(self.channel),      # 声道数
            '-f', 's16le',                # 输出格式为 16 位 PCM
            temp_pcm_path                 # 输出文件
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            with open(temp_pcm_path, 'rb') as f:
                pcm_data = f.read()
                
            # 删除临时文件
            os.remove(temp_pcm_path)
            
            return pcm_data
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 转换失败: {e}")
            raise
        except FileNotFoundError:
            print(f"找不到 FFmpeg 可执行文件: {self.ffmpeg_path}")
            raise

    def audio_to_opus(self, audio_file_path):
        """将音频文件转换为 Opus 格式，支持多种输入格式"""
        # 验证输入文件是否存在
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_file_path}")
        
        # 获取文件扩展名
        _, file_ext = os.path.splitext(audio_file_path)
        supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.aiff', '.au', '.mp4']
        
        if file_ext.lower() not in supported_formats:
            print(f"警告: 文件格式 {file_ext} 可能不被完全支持，尝试处理...")
        
        # 使用 FFmpeg 转换音频为 PCM
        raw_data = self.convert_to_pcm_with_ffmpeg(audio_file_path)
        
        # 获取音频信息
        audio = AudioSegment.from_file(audio_file_path)
        duration = len(audio)
        print(f"音频总时长: {duration}ms")
        print(f"音频格式: {file_ext}, 采样率: {audio.frame_rate}Hz, 声道数: {audio.channels}")

        print(f"音频PCM数据大小: {len(raw_data)}字节")

        # 初始化 Opus 编码器
        try:
            # 尝试使用正确的 API 初始化编码器
            encoder = opuslib_next.Encoder(
                self.opus_sample_rate, 
                self.opus_channel, 
                opuslib_next.OPUS_APPLICATION_AUDIO
            )
        except (TypeError, AttributeError):
            # 如果上面的初始化方式不正确，尝试其他方式
            try:
                # 尝试直接使用整数值代替 Application.AUDIO
                encoder = opuslib_next.Encoder(
                    self.opus_sample_rate, 
                    self.opus_channel, 
                    2049  # AUDIO 应用类型
                )
            except (TypeError, AttributeError):
                # 如果还是失败，尝试不传入应用类型参数，使用默认值
                encoder = opuslib_next.Encoder(
                    self.opus_sample_rate, 
                    self.opus_channel
                )
        
        frame_num = int(self.opus_sample_rate * self.opus_frame_time / 1000)
        frame_bytes_size = frame_num * self.opus_sample_width * self.opus_channel

        opus_datas = []

        for i in range(0, len(raw_data), frame_bytes_size):
            frame = raw_data[i:i + frame_bytes_size]
            if len(frame) < frame_bytes_size:
                # 用零填充不足的帧
                frame += b'\x00' * (frame_bytes_size - len(frame))

            # 将字节数据转换为 numpy 数组，然后编码
            np_frame = np.frombuffer(frame, dtype=np.int16)
            np_bytes = np_frame.tobytes()
            opus_data = encoder.encode(np_bytes, frame_num)
            opus_datas.append(opus_data)

        return opus_datas, duration
    
    def save_opus_to_file(self, opus_data_list, output_path):
        """将 Opus 数据保存到文件"""
        with open(output_path, 'wb') as f:
            # 写入 Ogg 封装头或其他需要的格式信息
            # 这里简单地将编码后的数据写入文件，实际应用中可能需要完整的 Opus 文件格式
            for opus_data in opus_data_list:
                f.write(opus_data)
    
    def save_opus_raw_custom(self, opus_datas, output_path):
        with open(output_path, 'wb') as f:
            for frame in opus_datas:
                f.write(len(frame).to_bytes(4, byteorder='big'))  # 写入帧长度
                f.write(frame)

    def load_opus_raw_custom(self, input_path):
        frames = []
        with open(input_path, 'rb') as f:
            data = f.read()
            index = 0
            while index < len(data):
                frame_length = int.from_bytes(data[index:index+4], byteorder='big')
                index += 4
                frame_data = data[index:index+frame_length]
                frames.append(frame_data)
                index += frame_length
        return frames
    
    def save_opus_raw(self, opus_data_list, output_path):
        """将原始 Opus 数据保存到文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(opus_data_list, f)

    def load_opus_raw(self, input_path):
        """从文件加载原始 Opus 数据"""
        with open(input_path, 'rb') as f:
            opus_data_list = pickle.load(f)
        return opus_data_list

    def opus_to_wav_file(self,output_file, opus_data : List[bytes] ) -> str:
        #output_file = "test.wav"
        decoder = opuslib_next.Decoder(self.opus_sample_rate, self.opus_channel)
        pcm_data = []
        for frame in opus_data:
            try:
                pcm_frame = decoder.decode(frame, self.opus_frame_size )
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                print(f"解码错误: {e}")
        
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(self.opus_channel)
            wav_file.setsampwidth(self.opus_sample_width)
            wav_file.setframerate(self.opus_sample_rate)
            #print(f"PCM 数据: {pcm_data}")
            #print(b''.join(pcm_data))
            wav_file.writeframes(b''.join(pcm_data))
        return output_file



if __name__ == "__main__":
    # 使用相对路径的 FFmpeg
    opus = Opus_Encoder()
    input_file = "test.mp3"  # 可以是 .mp3, .wav, .flac, .m4a 等
    opus_data, duration = opus.audio_to_opus(input_file)
    print(f"转换完成，共 {len(opus_data)} 个 Opus 数据包，音频时长 {duration}ms")

    # 保存到文件（可选）
    # opus.save_opus_to_file(opus_data, "output.opus")

    # print("原始 Opus 数据包示例:")
    # print(opus_data)
    # print("\n")

    opus.save_opus_raw_custom(opus_data, "test.opus")
    opus_data1 = opus.load_opus_raw_custom("test.opus")
    opus.opus_to_wav_file("test.wav", opus_data1)


'''
    print("原始 Opus 数据包示例:")
    print(opus_data)
    print("\n")

    opus.save_opus_raw(opus_data, "test.opus")
    opus_data = opus.load_opus_raw("test.opus")

    print("变换 Opus 数据包示例:")
    print(opus_data)
    print("\n")
'''
