import os
import pandas as pd
from PIL import Image
from model_utils import predict_image

def process_directory(directory_path, processor, model, device, threshold=0.3):
    """
    Processes all images in a directory and returns a DataFrame of results.
    """
    allowed_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    results_list = []
    
    if not os.path.exists(directory_path):
        return None, "Directory not found."

    files = [f for f in os.listdir(directory_path) if f.lower().endswith(allowed_extensions)]
    
    if not files:
        return None, "No supported image files found in the directory."

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            image = Image.open(file_path).convert("RGB")
            predictions = predict_image(image, processor, model, device, threshold)
            
            if predictions:
                top_pred = predictions[0]
                results_list.append({
                    "File Name": filename,
                    "Prediction": top_pred["label"],
                    "Confidence": round(top_pred["confidence"], 4)
                })
            else:
                results_list.append({
                    "File Name": filename,
                    "Prediction": "Model is not confident",
                    "Confidence": 0.0
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    df = pd.DataFrame(results_list)
    return df, None

def save_to_csv(df, output_path="batch_results.csv"):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return output_path
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return None
