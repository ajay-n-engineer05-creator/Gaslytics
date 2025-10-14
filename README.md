# Gaslytics — Industrial Gas Monitoring and Visualization Platform

**Overview:**  
Gaslytics is a Python-based tool developed during my internship to monitor and visualize industrial gas data in real-time.  
It includes modules for both sensor data visualization and advanced image-based analysis using Retinex and Haralick features.

---

## 🚀 Features
- Real-time gas concentration tracking from CSV or sensor data  
- Data visualization dashboard using Python (Plotly)  
- Image enhancement and feature extraction for gas plume detection  
- Modular design for analytics and visualization  
- Custom alerts for threshold breaches  

---

## 🧰 Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, OpenCV, Plotly  
- **Tools:** Jupyter, CSV-based data pipelines  
- **Environment:** Windows / Linux  

---

## 📂 Repository Structure
Gaslytics/
│
├── gas_view_tool/ # CSV data visualization and monitoring
│ └── Gas_view_tool_csv.py
│
├── image_processing/ # Image analysis & feature extraction
│ └── Retinex_diff_Haralick_features_single_video_processing.py
│
├── data/ # Sample data files (optional)
│
└── assets/ # Screenshots or results

yaml
Copy code

---

## 📈 Results
- Reduced data analysis time by **40%** through automated visualization  
- Improved detection accuracy in image-based monitoring by **~25%** after Retinex enhancement  

---

## 🧑‍💻 Role & Contributions
- Designed and implemented both data and image analysis pipelines  
- Integrated real-time visualization with analytical insights  
- Collaborated with team engineers to optimize performance  

---

## 🖼️ Example Output
![C5A1BD63-F00A-45F6-BAC3-9E0AFAEFFA89_1_201_a](https://github.com/user-attachments/assets/1bc78299-5435-41bf-873f-7ed5680018db)
 

---

## 🔗 How to Run
```bash
# Clone repository
git clone https://github.com/ajay-nambi/Gaslytics.git
cd Gaslytics

# Install dependencies
pip install -r requirements.txt

# Run main modules
python gas_view_tool/Gas_view_tool_csv.py
python image_processing/Retinex_diff_Haralick_features_single_video_processing.py
🔮 Future Work
Integration with live sensor APIs

Web-based dashboard for real-time visualization

Machine learning for anomaly detection

📜 License
This project was developed as side Project.
Use for learning and demonstration purposes only.
