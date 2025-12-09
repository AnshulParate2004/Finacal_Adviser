"""
LLM client wrapper utilities for NLP parser.
Uses Google Generative AI with LangChain for structured output.
"""
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient:
    """Wrapper for Google Generative AI LLM client."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the Google Generative AI model
            temperature: Temperature for generation (0.0 = deterministic)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in .env file or environment."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key
        )
    
    def invoke(self, prompt: str) -> str:
        """
        Invoke LLM with a simple prompt.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated response as string
        """
        response = self.llm.invoke(prompt)
        return response.content
    
    def invoke_with_parser(self, prompt: str, parser: PydanticOutputParser):
        """
        Invoke LLM with structured output using Pydantic parser.
        
        Args:
            prompt: Input prompt string
            parser: PydanticOutputParser for structured output
            
        Returns:
            Parsed Pydantic model instance
        """
        # Add format instructions to prompt
        format_instructions = parser.get_format_instructions()
        full_prompt = f"{prompt}\n\n{format_instructions}"
        
        # Get response
        response = self.llm.invoke(full_prompt)
        
        # Parse response
        parsed = parser.parse(response.content)
        return parsed


def get_llm_client(model_name: str = "gemini-1.5-flash", temperature: float = 0.0) -> LLMClient:
    """
    Factory function to get LLM client instance.
    
    Args:
        model_name: Name of the model
        temperature: Temperature for generation
        
    Returns:
        LLMClient instance
    """
    return LLMClient(model_name=model_name, temperature=temperature)
