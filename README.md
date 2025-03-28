# 🧠 Credit Card Churn Prediction

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dockerized](https://img.shields.io/badge/Docker-Ready-blue)
![Deployed on Azure](https://img.shields.io/badge/Deployed-Azure-blue)


This is an end-to-end machine learning project developed as a personal initiative to simulate a real-world business case and enhance my data science, MLOps, and cloud deployment skills.

The goal is to predict customer churn (i.e., whether a credit card user is likely to cancel their card) using a binomial classification model with over 90% accuracy.  
It includes **data exploration**, **feature engineering**, **model training**, and **deployment of a prediction API** using **Flask**, all containerized with **Docker**.

The solution is deployable both locally and in the cloud. It includes a working **CI/CD pipeline using GitHub Actions**, and has been successfully tested on:

- **Azure App Service** (Web App for Containers) via GitHub integration  
- **AWS EC2** (Docker-based deployment)

---

## 🔍 About this project

> ⚙️ This project was not developed for a specific company or production environment.  
> It was created to gain practical experience and demonstrate technical readiness to build deployable machine learning solutions.  
> All components follow industry-standard practices and are structured for clarity, scalability, and reproducibility.

---

## 🛠️ Tech Stack

- Python, pandas, scikit-learn
- Flask (for API)
- Docker
- GitHub Actions (CI/CD)
- Azure App Service (Web App for Containers)
- AWS EC2 (optional deployment)

---

## 🧩 Features

- Exploratory data analysis & pattern detection  
- Data transformation and feature engineering  
- ML model training and evaluation (>90% accuracy)  
- REST API for serving predictions  
- Docker containerization  
- CI/CD workflow with GitHub Actions  
- Cloud deployment: Azure & AWS

---

## 📅 Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
   - [Using Docker](#using-docker)
   - [Local Setup (Without Docker)](#local-setup-without-docker)
3. [Optional: Deployment on AWS EC2](#optional-deployment-on-aws-ec2)
4. [Environment Variables / GitHub Secrets](#environment-variables--github-secrets)
5. [Contributing](#contributing)
6. [License](#license)
7. [Maintainer / Author](#maintainer--author)

---

## 💡 Overview

- **Purpose:** Predict the likelihood of credit card churn using machine learning.  
- **Status:** Functional locally and in the cloud via Docker and GitHub Actions.

---

## 📁 Installation

### Using Docker

#### 1. Pull the Pre-Built Image
```bash
docker pull data5639/creditcardchurn-app-v0.0.1:latest
docker run -p 8080:8080 data5639/creditcardchurn-app-v0.0.1:latest
```

#### 2. Build the Image Locally
```bash
git clone https://github.com/dataismyname/credit_card_churn_mlproject.git
cd credit_card_churn_mlproject
docker build -t churn-prediction:local .
docker run -p 8080:8080 churn-prediction:local
```

Your application should now be accessible at [http://localhost:8080](http://localhost:8080).

#### 3. Test the API
```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"customer_id": 123, "balance": 1000, "transactions": 15}' \
    http://localhost:8080/predict
```

---

### Local Setup (Without Docker)

1. Install Python 3.8+
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
python application.py
```

---

## 🌐 Optional: Deployment on AWS EC2

Instructions for deploying the project on an AWS EC2 instance using Docker.

1. SSH into your EC2 instance.
2. Install Docker:
```bash
curl -fsSL https://get-docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```
3. Pull and run the Docker image:
```bash
docker pull data5639/creditcardchurn-app-v0.0.1:latest
docker run -d -p 8080:8080 data5639/creditcardchurn-app-v0.0.1:latest
```

---

## 🔐 Environment Variables / GitHub Secrets

If using GitHub Actions for deployment:

- `AZURE_WEBAPP_NAME`  
- `AZURE_CREDENTIALS` (from Azure service principal)  
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`  
- `ECR_REPOSITORY_NAME` (if using AWS ECR)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under MIT. See the [LICENSE](./LICENSE) file for details.

---

## 👤 Maintainer / Author

- **Name:** David A. Tirado A.  
- **Website:** [dataexpn.com](https://dataexpn.com)  
- **GitHub:** [@dataismyname](https://github.com/dataismyname)  
- **Email:** datirado@dataexpn.com

