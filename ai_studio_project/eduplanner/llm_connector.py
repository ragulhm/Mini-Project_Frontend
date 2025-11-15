# eduplanner/llm_connector.py (FINAL, CORRECTED VERSION)
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

class LLMConnector:
    """Handles all API calls to different LLM models via OpenRouter."""

    # --- Configuration ---
    # TIME_OUT: Set a maximum time (in seconds) the request is allowed to take before aborting.
    REQUEST_TIMEOUT = 90  
    
    # Model Mappings for the plan:
    # NOTE: "free" models can be unstable. Robust error handling is essential.
    # ----------------------------------------------------------------------
    # Content Generation (DeepSeek is the "expert")
    DEEPSEEK_MODEL = "meta-llama/llama-3-70b-instruct:free"
    
    # Agent Logic (OpenAI free model for structured tasks)
    OPENAI_AGENT_MODEL = "google/gemini-pro:free"
    # ----------------------------------------------------------------------
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    @staticmethod
    def _call_api(model: str, prompt: str, is_json: bool = False):
        """Internal method to handle the API request."""
        if not API_KEY:
            print("ERROR: OPENROUTER_API_KEY not set in .env file. Returning None.")
            return None

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Set response format to JSON if requested
        response_format = {"type": "json_object"} if is_json else None

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            
            "temperature": 0.7,
            "response_format": response_format,
            # Pass the timeout in the data payload as well (some APIs use this)
            "timeout": LLMConnector.REQUEST_TIMEOUT
        }

        try:
            # Send POST request, applying the timeout
            response = requests.post(
                LLMConnector.API_URL, 
                headers=headers, 
                json=data, 
                timeout=LLMConnector.REQUEST_TIMEOUT
            )
            
            # --- Check for API Errors (Non-200 Status Codes) ---
            if response.status_code != 200:
                print(f"ERROR: API returned status code {response.status_code} for model {model}.")
                print(f"DEBUG: Response text (first 200 chars): {response.text[:200]}...")
                return None
            
            # --- Successful 200 response, extract content ---
            response_json = response.json()
            
            # Safety check: ensure 'choices' exists
            if not response_json.get('choices'):
                 print(f"ERROR: API response missing 'choices' for model {model}. Full response: {response.text[:200]}...")
                 return None

            content = response_json['choices'][0]['message']['content']
            
            # --- Handle JSON Parsing if requested ---
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"ERROR: JSON Decode Error for model {model}. LLM returned non-JSON text. Content: {content[:200]}...")
                    return None
            
            return content
        
        except requests.exceptions.Timeout:
            print(f"ERROR: API request timed out after {LLMConnector.REQUEST_TIMEOUT}s for model {model}.")
            return None
        except requests.exceptions.RequestException as e:
            # Catches connection errors, DNS failure, 4xx/5xx status codes (less common with the 200 check above)
            print(f"CRITICAL API REQUEST ERROR for model {model}: {e}")
            return None
        except Exception as e:
            # General catch-all for any other unexpected error
            print(f"UNEXPECTED INTERNAL ERROR in _call_api for model {model}: {e}")
            return None

    @staticmethod
    def get_expert_response(prompt: str, is_json: bool = False):
        """Uses the DeepSeek model for content generation (Skill-Tree/Lesson Plan)."""
        return LLMConnector._call_api(
            model=LLMConnector.DEEPSEEK_MODEL, 
            prompt=prompt, 
            is_json=is_json
        )

    @staticmethod
    def get_agent_response(prompt: str, is_json: bool = False):
        """Uses the cost-effective OpenAI model for agent logic (Eval/Optimize/Analyze)."""
        return LLMConnector._call_api(
            model=LLMConnector.OPENAI_AGENT_MODEL, 
            prompt=prompt, 
            is_json=is_json
        )