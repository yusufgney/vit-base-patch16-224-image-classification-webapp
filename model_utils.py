import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F

# Model ID
MODEL_ID = "google/vit-base-patch16-224"

def load_model():
    """
    Loads the ViT processor and model.
    Uses GPU if available, otherwise CPU.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = ViTImageProcessor.from_pretrained(MODEL_ID)
        model = ViTForImageClassification.from_pretrained(MODEL_ID).to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def predict_image(image, processor, model, device, threshold=0.3):
    """
    Takes a PIL image and returns top-5 predictions.
    Filters results based on threshold.
    """
    try:
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Calculate probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top 5
        top5_prob, top5_indices = torch.topk(probs, 5)
        
        results = []
        for i in range(5):
            score = top5_prob[0][i].item()
            label = model.config.id2label[top5_indices[0][i].item()]
            
            if score >= threshold:
                results.append({"label": label, "confidence": score})
        
        return results
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []
