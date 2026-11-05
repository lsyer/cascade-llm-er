import os

def load_glm_key():
    with open(os.environ.get('ENV_FILE', '.env')) as f:
        for line in f:
            if line.startswith('LLM_API_KEY='):
                return line.split('=', 1)[1].strip()
    raise ValueError("API key not found")

GLM_URL = 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions'
QWEN_URL = 'http://www.netforger.com:8009/v1/chat/completions'
import os as _os
QWEN_KEY = _os.environ.get('QWEN_KEY', '')
GLM_KEY = load_glm_key()
