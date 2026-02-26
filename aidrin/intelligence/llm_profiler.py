import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMProfiler:
    """Uses an LLM (such as GPT-4o) to provide semantic data profiling and recommendations."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            
    def generate_actionable_insights(self, traditional_metrics: Dict[str, Any], ai_metrics: Dict[str, Any]) -> str:
        """Sends data metrics to LLM to get an action plan."""
        if not self.client:
            return "Cannot generate actionable insights: OPENAI_API_KEY is not set."
            
        prompt = f"""
        You are an expert AI Researcher analyzing a dataset's readiness for machine learning.
        Below are the calculated metrics:
        
        Traditional Quality Metrics:
        {json.dumps(traditional_metrics, indent=2)}
        
        AI-Specific Readiness Metrics:
        {json.dumps(ai_metrics, indent=2)}
        
        Provide a concise, highly actionable summary telling the data scientist exactly what they need to fix 
        (e.g., dropping columns with too many missing values, handling class imbalances, addressing bias).
        Output the response in markdown format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content or "No insights generated."
        except Exception as e:
            return f"Error during LLM insight generation: {e}"
