**🌕 Crater and Boulder Detection Project 🚀**

A YOLO-based web application for detecting craters and boulders from images and videos, with real-time visualization and downloadable results.

🔥 Features
✅ Upload images/videos for detection
✅ Real-time visualization of detected craters and boulders
✅ Download processed output with bounding boxes
✅ FastAPI backend for model inference
✅ Three.js frontend with a revolving 3D moon and interactive UI
✅ Display detection results directly on the frontend

⚙️ Tech Stack
Frontend: Three.js (without React), HTML, CSS, JavaScript

Backend: FastAPI (Python)

Model: YOLO (best.pt) trained for crater and boulder detection

Database: MongoDB (for potential result storage)
🎯 Model Performance
![image](https://github.com/user-attachments/assets/1d34e7e5-13ec-44fe-b627-7dc138a009ce)
**🚀 Installation**

1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/DilkhushYash/Crater-Boulder-Detection.git
cd Crater-Boulder-Detection
2️⃣ Install Dependencies
Backend
Make sure you have Python 3.10+ installed.

bash
Copy
Edit
cd backend  
pip install -r requirements.txt  
Install YOLO dependencies:

bash
Copy
Edit
pip install ultralytics
Frontend
Install a basic HTTP server (for development):

bash
Copy
Edit
cd frontend  
npm install  
3️⃣ Run the Project
✅ Start the backend server:

bash
Copy
Edit
cd backend  
uvicorn main:app --reload  
✅ Start the frontend:

bash
Copy
Edit
cd frontend  
npx http-server  
✅ Open your browser and visit: https://crater-boulder-detection.vercel.app/

📂 Project Structure
/backend  
 ├── main.py                  # FastAPI server  
 ├── model_loader.py           # YOLO model loading  
 ├── utils.py                  # Helper functions  
 ├── requirements.txt          # Backend dependencies  
 └── data/                     # Sample images/videos  
/frontend  
 ├── index.html                # Three.js UI  
 ├── style.css                 # Styling  
 ├── script.js                 # Detection logic and API calls  
 ├── moon.glb                  # 3D moon model  
 └── assets/                   # Team member images and other assets 

 Demo: 
 
![image](https://github.com/user-attachments/assets/d5aa973e-3ec1-4edf-8a2a-ac1536766e80)
![image](https://github.com/user-attachments/assets/a6f79ff0-39aa-41bf-a900-2e07a12aec2e)
![image](https://github.com/user-attachments/assets/b5dd0418-bb81-4e5e-a299-0f15c932d94a)
![image](https://github.com/user-attachments/assets/46d8a49f-e4e7-42d4-9e40-654dcf39d0e4)

**🛠️ To-Do**

 Improve the frontend UI/UX

 Add authentication for user uploads

 Store results in MongoDB for later retrieval

 Optimize model inference for faster processing
**
🤝 Contributors**

Devanshu Mangal (Team Lead, Backend & Model Integration)

Dilkhush Yash (Frontend & UI/UX)

 Dhairya Prajapati (Testing & Documentation)





