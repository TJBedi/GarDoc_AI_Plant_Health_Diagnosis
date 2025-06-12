import os
import json
import numpy as np
from flask import Flask, request, render_template, redirect, flash, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from PIL import Image
import logging
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'garddoc_secret_key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('garddoc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('garddoc')

# Load model and class labels
try:
    model = load_model("model/garddoc_resnet50_model.h5")
    with open("model/class_labels.json", "r") as f:
        class_labels = json.load(f)
    logger.info("Model and class labels loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or labels: {e}")
    model = None
    class_labels = {}

plant_care = {
    "apple": [
        "1. Requires 6–8 hours of full sunlight daily.",
        "2. Water deeply once a week; avoid waterlogging.",
        "3. Needs well-drained, slightly acidic soil (pH 6.0–6.5).",
        "4. Prune in late winter to remove dead or crowded branches.",
        "5. Use balanced fertilizer (10-10-10) in spring and midseason.",
        "6. Mulch around the base to retain moisture and prevent weeds."
    ],
    "brinjal": [
        "1. Needs 6–8 hours of direct sunlight.",
        "2. Keep soil moist but not soggy; water regularly.",
        "3. Prefers rich, well-drained soil with compost.",
        "4. Space plants 18–24 inches apart.",
        "5. Use stakes to support tall plants.",
        "6. Use neem oil to deter aphids and whiteflies."
    ],
    "lemon": [
        "1. Full sun required, at least 6 hours daily.",
        "2. Water deeply once or twice a week; allow soil to dry between watering.",
        "3. Requires slightly acidic, well-drained soil (pH 5.5–6.5).",
        "4. Use nitrogen-rich citrus fertilizer every 4–6 weeks.",
        "5. Protect from frost; ideal temperature is 20°C–30°C.",
        "6. Prune lightly to remove deadwood and improve airflow."
    ],
    "moneyplant": [
        "1. Bright, indirect light is ideal; avoid harsh sunlight.",
        "2. Water when top 1–2 inches of soil is dry.",
        "3. Trim regularly to encourage bushy growth.",
        "4. Use diluted liquid fertilizer once a month.",
        "5. Clean leaves regularly to improve air purification.",
        "6. Can grow in soil, water, or hydroponic containers."
    ],
    "neem": [
        "1. Needs full sun, minimum 6 hours per day.",
        "2. Water once weekly; allow soil to dry between watering.",
        "3. Tolerates poor soil; prefers well-drained, loamy soil.",
        "4. Apply compost or cow dung twice a year (spring and monsoon).",
        "5. Prune light branches to encourage new growth.",
        "6. Very drought-tolerant once mature."
    ],
    "potato": [
        "1. Requires full sunlight (6+ hours per day).",
        "2. Prefers loose, well-drained soil rich in organic matter.",
        "3. Provide 1–2 inches of water per week; keep soil evenly moist.",
        "4. Plant tubers 12 inches apart and hill soil up around stems.",
        "5. Mound soil regularly to protect tubers from sunlight.",
        "6. Harvest when plants flower and leaves turn yellow (approx. 70–120 days)."
    ],
    "tomato": [
        "1. Needs 6–8 hours of full sunlight daily.",
        "2. Water deeply at the base; avoid wetting the leaves.",
        "3. Use well-drained, fertile soil enriched with compost.",
        "4. Fertilize with phosphorus-rich feed every 2–3 weeks.",
        "5. Support plants with cages or stakes as they grow.",
        "6. Monitor for blight; use organic fungicides as needed."
    ],
    "tulsi": [
        "1. Requires 4–6 hours of direct sunlight daily.",
        "2. Water daily in summer and every alternate day in winter.",
        "3. Grows best in loamy soil with good drainage.",
        "4. Use compost tea or cow dung every 3–4 weeks.",
        "5. Pinch flower buds regularly to encourage leaf growth.",
        "6. Protect from frost or cold temperatures."
    ]
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(file_path):
    try:
        img = Image.open(file_path).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_class = class_labels[str(predicted_index)]
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")

        if confidence < 0.4:
            return "Unknown Plant", ["Confidence is too low for reliable classification. Try taking a clearer photo."]

        if "_healthy" in predicted_class:
            plant = predicted_class.replace("_healthy", "")
            status = "Healthy"
        elif "_diseased" in predicted_class:
            plant = predicted_class.replace("_diseased", "")
            status = "Diseased"
        else:
            plant = predicted_class
            status = "Identified"

        care_tips = plant_care.get(plant.lower(), ["Care instructions not available."])
        if status == "Diseased":
            care_tips.insert(0, "⚠️ This plant appears to have signs of disease. Consider treatment options.")

        label = f"{plant.capitalize()} ({status})"
        return label, care_tips

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Error in prediction", ["An error occurred during prediction. Please try again."]

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template("home.html")  # Your animated homepage

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            label, care_tips = predict_image(file_path)
            return render_template("index.html", prediction=label, care=True, care_tips=care_tips, image_file=filename)

        else:
            flash('Invalid file type.')
            return redirect(request.url)

    return render_template("index.html", prediction=None)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.errorhandler(413)
def too_large(error):
    flash('File too large. Max size is 16MB.')
    return redirect(url_for('predict')), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    flash('An unexpected error occurred.')
    return redirect(url_for('predict')), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

# Run the app
if __name__ == "__main__":
    
        logger.info("Starting GarDoc Flask server...")
        app.run(debug=True)
