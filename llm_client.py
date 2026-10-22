from functools import lru_cache
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

@lru_cache(maxsize=1)
def get_chat_model():
    config.check_mistral_api()
    return ChatMistralAI(
        model = config.MISTRAL_CHAT_MODEL,
        temperature = config.TEMPERATURE,
        api_key = config.MISTRAL_API_KEY
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    config.check_gemini_api()
    return GoogleGenerativeAIEmbeddings(
        model = config.EMBEDDING_MODEL,
        google_api_key = config.GEMINI_API_KEY
    )