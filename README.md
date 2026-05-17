🚀 Project Overview

Businesses generate massive amounts of customer transaction data every day.
Understanding customer behavior is essential for:

Personalized marketing
Customer retention
Revenue optimization
Loyalty programs
Business decision making

This project segments customers into meaningful groups using Machine Learning techniques and visual analytics.

🧠 Features

- Real-world Online Retail II dataset
- Data cleaning & preprocessing
- RFM (Recency, Frequency, Monetary) analysis
- Outlier detection & handling
- Feature scaling
- K-Means clustering
- Elbow Method for optimal K selection
- Silhouette Score evaluation
- PCA visualization
- Customer persona generation
- Interactive Streamlit dashboard
- AI-style business insights
- Download segmented customer data

📂 Dataset

Dataset Used: Online Retail II UCI Dataset

Transactions occurring between 2009 and 2011
Contains customer purchases from an online retail store

Dataset Source:

Kaggle Dataset Link

🛠️ Technologies Used
Category	Tools
Programming	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
Machine Learning	Scikit-learn
Dashboard	Streamlit
IDE	VS Code
📁 Project Structure
customer-segmentation/
│
├── data/
│   ├── online_retail.csv
│   ├── cleaned_online_retail.csv
│   ├── rfm_data.csv
│   └── customer_segments.csv
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── visuals/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
└── README.md
⚙️ Installation
1️⃣ Clone Repository
echo "# customer-segment" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/MalshaJayamanne/customer-segment.git
git push -u origin main

cd customer-segmentation
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run Jupyter Notebook
jupyter notebook

##### RUN APP ######
streamlit run app.py

Open:

notebooks/customer_segmentation.ipynb
🌐 Run Streamlit Dashboard
streamlit run app.py


Machine Learning Workflow

1. Data Cleaning
Removed missing Customer IDs
Removed duplicates
Removed negative quantities and prices
Converted date columns

2. Feature Engineering

Created:

Recency
Frequency
Monetary (RFM)

3. Data Preprocessing
Outlier handling using IQR
Standard scaling

4. Clustering

Applied:

K-Means Clustering

Used:

Elbow Method
Silhouette Score

5. Visualization
PCA customer visualization
RFM distribution analysis
Interactive dashboard charts