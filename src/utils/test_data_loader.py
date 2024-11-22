from data_loader import TalemaaderDataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_data_loader():
    loader = TalemaaderDataLoader()
    try:
        data = loader.prepare_evaluation_data()
        print(f"Successfully loaded {len(data['data'])} expressions")
        print("\nSample data:")
        print(data['data'][0])
        print(f"\nCorrect label: {data['correct_labels'][0]}")
    except Exception as e:
        print(f"Error testing data loader: {str(e)}")

if __name__ == "__main__":
    test_data_loader() 