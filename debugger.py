from src.utils import load_object

model = load_object("artifacts/model.pkl")
print(model)
print(dir(model))  # Check available attributes
