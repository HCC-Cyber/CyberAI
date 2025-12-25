import os
import asyncio
from asr.funasr.funasr_wrapper import FunASRWrapper
from utils.util import Util
from llm.chatglm import ChatGLM_LLM
from tts.edge import Edge_TTS
from audio_format.opus import Opus_Encoder

if __name__ == "__main__":
    
    # 获取配置文件
    config = Util.get_config()

    # 创建 LLM 对象
    llm = ChatGLM_LLM(config.get("LLM").get("ChatGLM"))
    resource = llm.generate_response("你好，你是谁？请用粤语回答")
    print(resource)

    # 创建 TTS 对象并合成语音
    tts = Edge_TTS(config.get("TTS").get("EdgeTTS"))
    output_file = asyncio.run(tts.text_to_speech(resource,  Util.get_process_dir() + "output" + os.sep + "test.mp3"))
    os.system(f'start {Util.get_process_dir() + "output" + os.sep + "test.mp3"}')

    # 创建 FunASR 对象并进行语音识别
    fun_asr = FunASRWrapper(config.get("ASR").get("FunASR"))
    resource = fun_asr.audio_file_to_text(Util.get_process_dir() + "output" + os.sep + "test.mp3")
    print(resource)

    # 将 mp3 转换为 opus 数据包
    opus = Opus_Encoder()
    input_file = Util.get_process_dir() + "output" + os.sep + "test.mp3"  # 可以是 .mp3, .wav, .flac, .m4a 等
    opus_data, duration = opus.audio_to_opus(input_file)
    print(f"转换完成，共 {len(opus_data)} 个 Opus 数据包，音频时长 {duration}ms")
    opus.save_opus_raw_custom(opus_data,  Util.get_process_dir() + "output" + os.sep + "test.opus")

    # 将 opus 数据包转换回 wav 文件
    # opus.opus_to_wav_file(Util.get_process_dir() + "output" + os.sep + "test.wav", opus.load_opus_raw_custom(Util.get_process_dir() + "output" + os.sep + "test.opus"))

    # 使用 FunASR 识别 opus 数据包内容
    resource = fun_asr.opus_data_to_text(opus_data, Util.get_process_dir() + "output" + os.sep + "test.wav")
    #resource = fun_asr.opus_data_to_text(opus_data, Util.get_random_file_path(Util.get_process_dir(), "wav"))

    # os.system("output.mp3")
    print("main:" + resource)
