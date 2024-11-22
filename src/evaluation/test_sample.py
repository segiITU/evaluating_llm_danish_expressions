import logging
from src.utils.data_loader import TalemaaderDataLoader
from src.models.gpt import GPTModel

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sample_predictions(n_samples: int = 5):
    """Test GPT model on first few items from dataset."""
    print(f"\nTesting {n_samples} samples...")
    
    # Load data
    loader = TalemaaderDataLoader()
    test_df = loader.load_evaluation_data()
    sample_df = test_df.head(n_samples)
    
    # Initialize model
    model = GPTModel()
    
    # Process each idiom
    for idx, row in sample_df.iterrows():
        print(f"\nTesting idiom {idx + 1}/{n_samples}")
        print(f"Idiom: {row['talemaade_udtryk']}")
        
        options = {
            'A': row['A'],
            'B': row['B'],
            'C': row['C'],
            'D': row['D']
        }
        
        try:
            prediction = model.predict(row['talemaade_udtryk'], options)
            print(f"Options:")
            for letter, definition in options.items():
                print(f"{letter}: {definition}")
            print(f"Model prediction: {prediction}")
            print(f"Predicted definition: {options[prediction]}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_sample_predictions() 