# CyberAI 系统架构文档

## 1. 项目概述

CyberAI 是一个集成语音识别（ASR）、大语言模型（LLM）、文本转语音（TTS）和音频格式转换功能的AI系统。该项目支持完整的语音对话流程：语音输入→语音识别→语言模型处理→文本转语音→语音输出，并支持多种音频格式转换。

## 2. 项目结构

```
CyberAI/
├── ai_core/                 # 核心功能模块
│   ├── asr/                # 语音识别模块
│   │   └── funasr/         # FunASR实现
│   │       └── funasr_wrapper.py
│   ├── audio_format/       # 音频格式转换模块
│   │   └── opus.py         # Opus编码解码实现
│   ├── llm/                # 大语言模型模块
│   │   └── chatglm.py      # ChatGLM实现
│   ├── tts/                # 文本转语音模块
│   │   └── edge.py         # Edge TTS实现
│   ├── utils/              # 工具类
│   │   └── util.py         # 通用工具
│   ├── main.py             # 程序入口
│   ├── model_download.py   # 模型下载器
│   └── readme/             # 说明文档
├── config.yaml             # 配置文件
├── config.txt              # 配置文件
└── README.md               # 项目说明
```

## 3. 模块功能说明

### 3.1 语音识别模块 (ASR)
- **文件路径**: `ai_core/asr/funasr/funasr_wrapper.py`
- **功能**: 使用FunASR进行语音识别，支持音频文件识别和Opus数据包识别
- **核心类**: [FunASRWrapper](file:CyberAI/ai_core/asr/funasr/funasr_wrapper.py#L7-L52)
- **主要方法**:
  - [audio_file_to_text(audio_file_path)](file:CyberAI/ai_core/asr/funasr/funasr_wrapper.py#L39-L46): 将音频文件转换为文本
  - [opus_data_to_text(opus_data, audio_file_path)](file:CyberAI/ai_core/asr/funasr/funasr_wrapper.py#L48-L52): 将Opus数据包转换为文本

### 3.2 大语言模型模块 (LLM)
- **文件路径**: `ai_core/llm/chatglm.py`
- **功能**: 使用ChatGLM大语言模型进行对话和文本生成
- **核心类**: [ChatGLM_LLM](file:CyberAI/ai_core/llm/chatglm.py#L2-L19)
- **主要方法**:
  - [generate_response(user_input)](file:CyberAI/ai_core/llm/chatglm.py#L10-L19): 根据用户输入生成响应

### 3.3 文本转语音模块 (TTS)
- **文件路径**: `ai_core/tts/edge.py`
- **功能**: 使用Edge TTS将文本转换为语音
- **核心类**: [Edge_TTS](file:CyberAI/ai_core/tts/edge.py#L2-L12)
- **主要方法**:
  - [text_to_speech(text, output_file)](file:CyberAI/ai_core/tts/edge.py#L6-L12): 异步将文本转换为语音并保存到文件

### 3.4 音频格式转换模块 (Audio Format)
- **文件路径**: `ai_core/audio_format/opus.py`
- **功能**: 实现音频格式转换，特别是Opus编码解码
- **核心类**: [Opus_Encoder](file:CyberAI/ai_core/audio_format/opus.py#L10-L200)
- **主要方法**:
  - [audio_to_opus(audio_file_path)](file:CyberAI/ai_core/audio_format/opus.py#L77-L142): 将音频文件转换为Opus数据包
  - [opus_to_wav_file(output_file, opus_data)](file:CyberAI/ai_core/audio_format/opus.py#L182-L200): 将Opus数据包转换为WAV文件
  - [save_opus_raw_custom(opus_datas, output_path)](file:CyberAI/ai_core/audio_format/opus.py#L152-L156): 保存Opus数据到文件
  - [load_opus_raw_custom(input_path)](file:CyberAI/ai_core/audio_format/opus.py#L158-L169): 从文件加载Opus数据

### 3.5 工具模块 (Utils)
- **文件路径**: `ai_core/utils/util.py`
- **功能**: 提供通用工具方法
- **核心类**: [Util](file:CyberAI/ai_core/utils/util.py#L9-L62)
- **主要方法**:
  - [get_process_dir()](file:CyberAI/ai_core/utils/util.py#L11-L14): 获取项目根目录
  - [get_config()](file:CyberAI/ai_core/utils/util.py#L26-L35): 获取配置
  - [get_random_file_path(dir, ex_name)](file:CyberAI/ai_core/utils/util.py#L58-L62): 生成随机文件路径

### 3.6 模型下载模块 (Model Download)
- **文件路径**: `ai_core/model_download.py`
- **功能**: 下载和管理AI模型
- **核心类**: [ModelDownloader](file:CyberAI/ai_core/model_download.py#L14-L184)
- **主要方法**:
  - [download(force_download=False)](file:CyberAI/ai_core/model_download.py#L76-L118): 下载模型
  - [check_model_exists()](file:CyberAI/ai_core/model_download.py#L203-L215): 检查模型是否存在

## 4. 配置文件

### 4.1 config.yaml
```yaml
TTS:
  EdgeTTS:
    type: "EdgeTTS"
    voice: "zh-CN-XiaoxiaoNeural"

LLM:
  ChatGLM:
    type: "ChatGLM"
    model_name: "glm-4.6v-flash" 
    api_key: "your_api_key"
    url: "https://open.bigmodel.cn/api/paas/v4" 

ASR:
  FunASR:
    type: "FunASRWrapper"
    output_dir: "ai_core/asr/funasr/temp"
```

## 5. 系统接口定义

### 5.1 ASR接口
```python
class ASRInterface:
    def __init__(self, config):
        """
        初始化ASR接口
        :param config: 配置对象，包含ASR相关配置
        """
        pass

    def audio_file_to_text(self, audio_file_path):
        """
        将音频文件转换为文本
        :param audio_file_path: 音频文件路径
        :return: 识别的文本
        """
        pass

    def opus_data_to_text(self, opus_data, audio_file_path):
        """
        将Opus数据转换为文本
        :param opus_data: Opus数据包列表
        :param audio_file_path: 用于临时保存解码音频的路径
        :return: 识别的文本
        """
        pass
```

### 5.2 LLM接口
```python
class LLMInterface:
    def __init__(self, config):
        """
        初始化LLM接口
        :param config: 配置对象，包含LLM相关配置
        """
        pass

    def generate_response(self, user_input):
        """
        生成对用户输入的响应
        :param user_input: 用户输入的文本
        :return: 生成的响应文本
        """
        pass
```

### 5.3 TTS接口
```python
class TTSInterface:
    def __init__(self, config):
        """
        初始化TTS接口
        :param config: 配置对象，包含TTS相关配置
        """
        pass

    async def text_to_speech(self, text, output_file):
        """
        将文本转换为语音
        :param text: 要转换的文本
        :param output_file: 输出音频文件路径
        :return: 输出文件路径
        """
        pass
```

### 5.4 Audio Format接口
```python
class AudioFormatInterface:
    def audio_to_opus(self, audio_file_path):
        """
        将音频文件转换为Opus数据包
        :param audio_file_path: 音频文件路径
        :return: (Opus数据包列表, 音频时长)
        """
        pass

    def opus_to_wav_file(self, output_file, opus_data):
        """
        将Opus数据包转换为WAV文件
        :param output_file: 输出WAV文件路径
        :param opus_data: Opus数据包列表
        :return: 输出文件路径
        """
        pass

    def save_opus_raw_custom(self, opus_datas, output_path):
        """
        保存Opus数据到文件
        :param opus_datas: Opus数据包列表
        :param output_path: 输出文件路径
        """
        pass

    def load_opus_raw_custom(self, input_path):
        """
        从文件加载Opus数据
        :param input_path: 输入文件路径
        :return: Opus数据包列表
        """
        pass
```

## 6. 主要流程

### 6.1 对话流程
1. 使用LLM生成响应文本
2. 使用TTS将文本转换为语音
3. 使用ASR识别语音输入
4. 使用Audio Format进行音频格式转换

### 6.2 音频格式转换流程
1. 使用FFmpeg将音频文件转换为PCM格式
2. 使用Opus编码器将PCM数据编码为Opus数据包
3. 支持将Opus数据包解码回WAV格式

## 7. 技术依赖

- `openai`: 用于LLM接口
- `edge_tts`: 用于TTS功能
- `funasr`: 用于ASR功能
- `opuslib_next`: 用于Opus编码解码
- `pydub`: 用于音频处理
- `numpy`: 用于数据处理
- `ffmpeg`: 用于音频格式转换

## 8. 扩展性考虑

1. **模块化设计**: 各功能模块独立，易于扩展和替换
2. **配置驱动**: 通过配置文件可以灵活切换不同实现
3. **接口抽象**: 定义了清晰的接口，便于实现不同的后端
4. **单例模式**: 在音频格式转换模块中使用单例模式，确保资源复用

这份文档提供了项目的完整架构和接口定义，任何AI助手都可以根据这份文档来实现相应功能，确保代码的一致性和可维护性。