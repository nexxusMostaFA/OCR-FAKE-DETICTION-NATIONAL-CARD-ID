# OCR-FAKE-DETICTION-NATIONAL-CARD-ID


Face Recognition API with InsightFace + Flask + MongoDB
A powerful and modular face recognition API built with Flask, InsightFace’s AuraFace model, and MongoDB. It allows users to sign up using their face and later verify identity securely through image uploads.

This API includes advanced facial quality checks, face covering detection (like masks or sunglasses), and strict validation logic to ensure biometric accuracy and integrity.

🚀 Features
🎯 Single Face Detection: Rejects multi-face or no-face images

😷 Face Covering Detection: Detects masks, sunglasses, or occlusions

📐 Face Quality Check: Validates size, clarity, tilt, skin tone & landmark visibility

🔁 Duplicate Prevention: Embedding comparison to avoid re-registration

🔐 Face Verification: Matches new images with stored embeddings

🧠 AuraFace Model: Loaded via HuggingFace + InsightFace

📦 MongoDB Integration: Stores normalized embeddings using PyMongo

🧾 Detailed Logging: For debugging, error handling, and traceability

🧰 Tech Stack
Layer	Technology
Backend	Flask (REST API)
Face Model	AuraFace via InsightFace
Embedding DB	MongoDB (via PyMongo)
Image Processing	OpenCV, NumPy
Serialization	Pickle + BSON Binary
Deployment	Flask CLI or custom args

🏗️ Project Structure
bash
Copy
Edit
.
├── app.py                      # Main Flask application
├── models/auraface/            # Downloaded model from HuggingFace
├── uploads/                    # Temp folder for incoming images
├── .env                        # Environment file for MongoDB URI
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
⚙️ Setup Instructions
1. Clone the repository
bash
Copy
Edit
git clone  https://github.com/nexxusMostaFA/OCR-FAKE-DETICTION-NATIONAL-CARD-ID.git
cd face-recognition-api
2. Create & activate virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Set up environment variables
Create a .env file and add your MongoDB connection string:

env
Copy
Edit
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/
🚦 Running the Server
Option 1: Flask CLI
bash
Copy
Edit
python app.py
Option 2: With Custom Args
bash
Copy
Edit
python app.py --host 0.0.0.0 --port 7000 --debug
📮 API Endpoints
GET /
Check if API is running.

Response:

json
Copy
Edit
{ "status": "success", "message": "Face Recognition API is running" }
POST /signUp
Register a new user by uploading a clear face image.

Form-data:

file: (Required) Image file (jpg/jpeg/png)

Response (Success):

json
Copy
Edit
{
  "status": "success",
  "message": "Face stored successfully with user_id: <uuid>"
}
Response (Error):

json
Copy
Edit
{
  "status": "error",
  "message": "This face already exists in the database"
}
POST /verify
Verify a face image against registered faces.

Form-data:

file: (Required) Image file

threshold: (Optional) Similarity threshold (default = 0.5)

Response:

json
Copy
Edit
{
  "status": "success",
  "message": "Face verified successfully with confidence: 0.94",
  "verified": true,
  "user_id": "123e4567-e89b-12d3-a456-426614174000"
}
🔍 Face Validation Logic
Each uploaded image is passed through:

Face Detection

Face Quality Check

Face Covering Detection

Duplicate Detection

Embedding Generation & Comparison

 

 

