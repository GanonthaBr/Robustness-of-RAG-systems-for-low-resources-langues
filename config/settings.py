"""Configuration settings for AFRI-RAG"""

# Languages supported
LANGUAGES = ['en', 'swa', 'yor', 'kin']

# Language to Wikipedia dataset mapping
WIKIPEDIA_DATASETS = {
    'en': ('wikimedia/wikipedia', '20231101.en'),
    'swa': ('wikimedia/wikipedia', '20231101.sw'),
    'yor': ('wikimedia/wikipedia', '20231101.yo'),
    'kin': ('wikimedia/wikipedia', '20231101.rw'),
}

# Language to prompt template mapping
PROMPT_TEMPLATES = {
    'en': """Documents:\n{context}\n\nQuestion: {question}\nAnswer concisely:""",
    
    'swa': """Nyorota hati zifuatazo kisha ujibu swali.\n\n{context}\n\nSwali: {question}\nJibu kwa Kiswahili kwa ufupi:""",
    
    'yor': """Àwọn ìwé wọ̀nyí:\n\n{context}\n\nÌbéèrè: {question}\nDáhùn ní Èdè Yorùbá:""",
    
    'kin': """Inyandiko zikurikira:\n\n{context}\n\nIkibazo: {question}\nIgisubizo mu Kinyarwanda:""",
}

# Retrieval settings
RETRIEVAL_K = 10
RAG_K_BY_LANGUAGE = {
    'swa': 10,
    'yor': 3,
    'kin': 5,
}

# Embedding models for comparison
EMBEDDING_MODELS = {
    'e5-base': 'intfloat/multilingual-e5-base',
    'qwen3': 'Qwen/Qwen3-Embedding-8B',
}

# Default embedding model
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Generation settings
LLM_MODELS = {
    'afriqueqwen-8b': 'McGill-NLP/AfriqueQwen-8B',
    'qwen2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
}

# Default LLM model
LLM_MODEL = "McGill-NLP/AfriqueQwen-8B"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Dataset settings
AFRIQA_LANGUAGES = ['swa', 'yor', 'kin', 'en']
IROKO_LANGUAGES = ['swa', 'yor']  # IrokoBench doesn't have Kinyarwanda