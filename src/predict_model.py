import sys
sys.path.append("..")
from train_model import TumorAnalysisModel
from PIL import Image
from utils import CLA_label

model_params={
    "shape_in": (3,256,256), 
    "initial_filters": 8,    
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2}

if __name__ == "__main__":
    model = TumorAnalysisModel("small")
    model.load_model("../weights.pt",model_params)
    model.model.eval()
    prediction, probability = model.predict(Image.open("../pred_examples/healthy.jpg").convert('RGB'))
    print(CLA_label[prediction], probability.numpy()[0][prediction])