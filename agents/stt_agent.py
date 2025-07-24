import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from openai import OpenAI, AzureOpenAI
from instructions.stt_instructions import instructions
import os

from dotenv import load_dotenv
load_dotenv()

openai_api_version = os.environ.get("OPENAI_API_VERSION")
llm_4o_mini_azure_endpoint = os.environ.get("LLM_AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")



class STTValidator:
    def __init__(self):
        """
        Initialize STT Validator
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use for validation
        """
        self.client = AzureOpenAI(api_version=openai_api_version, azure_endpoint=llm_4o_mini_azure_endpoint, api_key=azure_openai_api_key)
        self.validation_history = []
        
    def validate_message(self, stt_output: str) -> str:
        """
        Validate STT output using GPT API
        
        Args:
            stt_output: Raw output from Speech-to-Text
            
        Returns:
            Validated message string
        """
        try:
            # Call GPT API for validation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": stt_output}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_content = response.choices[0].message.content
            validation_result = json.loads(response_content)
            
            validated_message = validation_result.get('validated_user_message', '')
            
            # Handle invalid input case
            if validated_message.lower() == 'invalid_input':
                final_message = "I didn't get what you said"
            else:
                final_message = validated_message
            
            # Store the validation data in class
            self._store_validation(stt_output, validated_message, final_message)
            
            return final_message
            
        except json.JSONDecodeError:
            print(f"JSON Decode Error, result: {response_content}")
            return "I didn't get what you said"
            
        except Exception as e:
            print(f"Error in STT Validator: {e}")
            return "I didn't get what you said"
    
    def _store_validation(self, original: str, validated: str, final: str):
        """
        Store validation data in class instance
        
        Args:
            original: Original STT output
            validated: Validated message from GPT
            final: Final message returned to user
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'original_stt': original,
            'validated_message': validated,
            'final_message': final
        }
        
        self.validation_history.append(entry)
    
    def get_validation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve stored validation history
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            List of validation entries
        """
        if limit:
            return self.validation_history[-limit:]
        return self.validation_history
    
    def clear_history(self):
        """Clear all stored validation history"""
        self.validation_history = []


# #Example usage
# if __name__ == "__main__":
#     # Initialize validator
#     validator = STTValidator()
    
#     # Validate some STT output
#     while True: 
#         stt_text = input("Enter ")
#         validated_result = validator.validate_message(stt_text)
#         print(f"Validated message: {validated_result}")
    
#         # Get validation history
#         history = validator.get_validation_history(limit=5)
#         print(f"Recent validations: {len(history)}")