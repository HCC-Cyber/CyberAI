import edge_tts

class Edge_TTS:
    def __init__(self, config):
        self.voice = config.get("voice")

    async def text_to_speech(self, text, output_file):
        communicate = edge_tts.Communicate(
            text,
            self.voice,
        )
        await communicate.save(output_file)
        return output_file