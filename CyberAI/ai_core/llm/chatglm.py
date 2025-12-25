from openai import OpenAI

class ChatGLM_LLM:
    def __init__(self, config):

        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        self.url = config.get("url")
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)

    def generate_response(self, user_input):
        dialogue = [
        {"role": "system", "content": "你是一个台湾女孩"},
        {"role": "user", "content": user_input},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=dialogue,
        )
        return response.choices[0].message.content

