import google.generativeai as genai
from typing import Dict, List
import os
from src.models.base_model import BaseModel, retry_on_error
from src.config.model_configs import MODEL_CONFIGS
import time
from tqdm import tqdm

class GeminiModel(BaseModel):
    def __init__(self, model_config_key: str = "gemini"):
        super().__init__(model_config_key)
        # Configure the Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.config = MODEL_CONFIGS[model_config_key]
        self.model = genai.GenerativeModel(self.config["model_name"])
        
    @retry_on_error(max_retries=3)
    def predict(self, expression: str, definitions: Dict[str, str]) -> str:
        prompt = self.format_prompt(expression, definitions)
        start_time = time.time()
        
        try:
            self.api_stats["total_requests"] += 1
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config["temperature"],
                    max_output_tokens=self.config.get("max_output_tokens", 1000),
                    candidate_count=1
                )
            )
            
            # Update API stats
            latency = time.time() - start_time
            self.api_stats["average_latency"] = (
                (self.api_stats["average_latency"] * (self.api_stats["total_requests"] - 1) + latency)
                / self.api_stats["total_requests"]
            )
            
            prediction = response.text.strip()
            if not self.validate_prediction(prediction):
                self.logger.warning(f"Invalid prediction format: {prediction}")
                return None
                
            return prediction
            
        except Exception as e:
            self.api_stats["failed_requests"] += 1
            self.logger.error(f"Prediction error: {str(e)}")
            raise
            
    def batch_predict(self, data: List[Dict[str, str]], batch_size: int = 10) -> List[str]:
        predictions = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            batch_predictions = []
            
            for item in batch:
                prediction = self.predict(item['expression'], item)
                batch_predictions.append(prediction)
                time.sleep(1)  # Rate limiting
                
            predictions.extend(batch_predictions)
            
        # Save metrics after batch processing
        self.save_metrics()
        return predictions

if __name__ == "__main__":
    test_data = {
        "expression": "have sommerfugle i maven",
        "definition_a": "føle sig dårligt tilpas",
        "definition_b": "have spist for meget",
        "definition_c": "være sulten",
        "definition_d": "være nervøs eller anspændt"
    }
    
    model = GeminiModel()
    prediction = model.predict(test_data["expression"], test_data)
    print(f"Prediction: {prediction}")
