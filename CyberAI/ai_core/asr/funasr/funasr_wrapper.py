import os
from funasr import AutoModel  # 现在能正确导入真正的funasr库
from funasr.utils.postprocess_utils import rich_transcription_postprocess 
import torch

from audio_format.opus import Opus_Encoder

class FunASRWrapper:  # 建议也修改类名
    def __init__(self,config):
        self.dir = config.get("output_dir")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.current_dir, "model")
        
        self.model = AutoModel(
            model=self.model_dir,
            vad_kwargs={"max_single_segment_duration": 30000},
            # device = "cuda:0" if torch.cuda.is_available() else "cpu",
            disable_update=True,
            hub="hf",
        )
        self.raw_outputs = {}
        self.hooks = []

    def _register_hook(self):
        output_layer = self.model.model.ctc.ctc_lo

        def hook_function(module, input, output):
            self.raw_outputs["ctc_logits"] = output.clone()

        hook = output_layer.register_forward_hook(hook_function)
        self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.raw_outputs.clear()


    def audio_file_to_text(self, audio_file_path):
        res = self.model.generate(
            input=audio_file_path,
            language="auto",
            use_itn=True,
        )
        # return res[0]["text"]
        return rich_transcription_postprocess(res[0]["text"])

    def opus_data_to_text(self,opus_data , audio_file_path):
        opus = Opus_Encoder()
        opus.opus_to_wav_file(audio_file_path, opus_data)
        res = self.audio_file_to_text(audio_file_path)
        return res

if __name__ == "__main__":
    funasr = FunASRWrapper()
    funasr._register_hook()
    res = funasr.audio_file_to_text("test.wav")

    if funasr.raw_outputs:  # 获取原始输出
        logits = funasr.raw_outputs.get("ctc_logits")
        print("CTC Logits Shape:", logits.shape)

        token_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
        print("Token IDs:", token_ids)


    print(res)