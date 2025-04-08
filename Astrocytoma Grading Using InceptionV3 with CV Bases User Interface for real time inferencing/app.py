from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
from datetime import datetime
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')  # FIXED: absolute path for saving
RELATIVE_UPLOAD_PATH = 'uploads'  # for url_for
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(weights=None, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("InceptionV3_astrocytoma_20_epoch.pth", map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

labels = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    results_table = []

    # Clear previously uploaded images
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file:
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)  # FIXED: actually saving to disk

                # Predict
                image = Image.open(save_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_probs, top_idxs = probs.topk(3)

                    top_preds = [(labels[i], f"{p.item()*100:.2f}%") for i, p in zip(top_idxs[0], top_probs[0])]
                    predictions.append((filename, top_preds, f"{RELATIVE_UPLOAD_PATH}/{filename}"))

                    results_table.append({
                        "Filename": filename,
                        "Top 1": f"{top_preds[0][0]} ({top_preds[0][1]})",
                        "Top 2": f"{top_preds[1][0]} ({top_preds[1][1]})",
                        "Top 3": f"{top_preds[2][0]} ({top_preds[2][1]})",
                        "Datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

        # Save results to CSV
        csv_path = os.path.join(app.root_path, 'static', 'predictions.csv')
        pd.DataFrame(results_table).to_csv(csv_path, index=False)
        
    return render_template('index.html', predictions=predictions, table=results_table)

if __name__ == '__main__':
    app.run(debug=True)
