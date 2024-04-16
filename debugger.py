#import dill  # or pickle, depending on what you're using
from src.utils import load_object
import numpy as np

# Load
loaded_le = load_object('artifacts/label_encoder.pkl')

# Test
sample_labels = [1.]  # Adjust sample values to actual labels used
int_array = [int(num) for num in sample_labels]
#int_array = int(sample_labels)
#sample_labels = np.array(sample_labels)
print(int_array)
print(type(int_array))
sample_categorical = loaded_le.inverse_transform(int_array)
print(f"Decoded labels: {sample_categorical}")
