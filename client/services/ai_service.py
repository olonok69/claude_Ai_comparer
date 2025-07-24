import json
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from typing import Optional
from config import MODEL_OPTIONS

# Load environment variables
load_dotenv()

def create_llm_model(llm_provider: str, **kwargs):
    """Create a language model based on the selected provider."""
    
    if llm_provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OpenAI API key not found in environment variables")
            st.stop()
            
        return ChatOpenAI(
            openai_api_key=api_key,
            model=MODEL_OPTIONS['OpenAI'],
            temperature=kwargs.get('temperature', 0.7),
        )
        
    elif llm_provider == "Azure OpenAI":
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_ENDPOINT")
        azure_deployment = os.getenv("AZURE_DEPLOYMENT")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        
        if not all([azure_api_key, azure_endpoint, azure_deployment, azure_api_version]):
            st.error("❌ Azure OpenAI configuration incomplete. Please check your environment variables.")
            st.stop()
            
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            azure_deployment=azure_deployment,
            api_version=azure_api_version,
            temperature=kwargs.get('temperature', 0.7),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    

def get_response(prompt: str, llm_provider: str):
    """Get a response from the LLM using the standard LangChain interface."""
    try:
        # Create the LLM instance dynamically
        llm = create_llm_model(llm_provider)

        # Wrap prompt in a HumanMessage
        message = HumanMessage(content=prompt)

        # Invoke model and return the output content
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return f"Error during LLM invocation: {str(e)}"

def get_response_stream(
    prompt: str,
    llm_provider: str,
    system: Optional[str] = '',
    temperature: float = 1.0,
    max_tokens: int = 4096,
    **kwargs,
    ):
    """
    Get a streaming response from the selected LLM provider.
    All provider-specific connection/auth should be handled via environment variables.
    """
    try:
        # Add streaming and generation params to kwargs
        kwargs.update({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": True
        })

        # Create the LLM with streaming enabled
        llm = create_llm_model(llm_provider, **kwargs)

        # Compose messages
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        # Stream the response
        stream_response = llm.stream(messages)
        return stream_response
    except Exception as e:
        st.error(f"[Error during streaming: {str(e)}]")
        st.stop()