# AI-FASHION-STYLIST 
🧥 AI Fashion Stylist

An AI-powered fashion recommendation web app that analyzes a user-uploaded photo to detect gender, extract dominant outfit colors, and provide personalized outfit suggestions for Party, Formal, and Casual occasions.

Built using Python, Streamlit, OpenCV, DeepFace, and Machine Learning.

🚀 Features

📷 Upload a photo (JPG / PNG)

🧠 AI-based gender detection using DeepFace

🎨 Color palette extraction from clothing using K-Means clustering

👗 Smart outfit suggestions for:

Party

Formal

Casual

🎯 Complementary color recommendations

⚡ Fast, interactive Streamlit UI

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python

Computer Vision: OpenCV

Machine Learning: Scikit-learn (KMeans)

AI Face Analysis: DeepFace

Image Processing: Pillow, NumPy

📂 Project Structure
AI-Fashion-Stylist/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Required Python libraries
└── README.md            # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-fashion-stylist.git
cd ai-fashion-stylist

2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


Then open the browser at:

http://localhost:8501

📸 How It Works

User uploads a clear photo

App detects gender using DeepFace

Clothing region is analyzed for dominant colors

A color palette is generated

Outfit suggestions are displayed based on:

Detected gender

Extracted colors

Occasion type

📌 Example Output

🎨 Extracted color palette from outfit

👕 Lead & complementary color suggestions

👗 Occasion-wise fashion tips  
