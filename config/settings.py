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
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# Generation settings
LLM_MODEL = "McGill-NLP/AfriqueQwen-8B"  # or -1.5B for testing
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7

# Dataset settings
AFRIQA_LANGUAGES = ['swa', 'yor', 'kin', 'en']
IROKO_LANGUAGES = ['swa', 'yor']  # IrokoBench doesn't have Kinyarwanda