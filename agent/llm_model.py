from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import torch
import os

def load_llm():
    # Load environment variables (for Hugging Face token)
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_auth_token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Create a text generation pipeline
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )

    # Wrap the pipeline for LangChain compatibility
    llm = HuggingFacePipeline(pipeline=text_gen)
    return llm
