# MJIC AI Health Monitoring Solution

**AI-Powered Diabetes Risk Assessment & Health Monitoring Kiosk with Clinical Report Analysis**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Core Modules](#core-modules)
- [Dataset Requirements](#dataset-requirements)
- [Model Details](#model-details)
- [Usage Examples](#usage-examples)
- [AI Integration](#ai-integration)
- [Future Enhancements](#future-enhancements)

---

## Overview

The MJIC AI Health Monitoring Solution is a comprehensive web-based health assessment platform that combines machine learning, clinical guideline integration, and AI-powered recommendations. The system supports two primary use cases:

1. **Diabetes Risk Assessment**: Analyzes clinical PDF reports to extract patient vital signs and medical history, predicting diabetes risk using a pre-trained ANN model
2. **Real-time Health Monitoring**: Direct vital sign input through an interactive kiosk interface for immediate multi-condition health assessment

The platform integrates **Ollama LLM** for generating personalized health recommendations and follows **Malaysian clinical guidelines** for health benchmarking and risk assessment.

---

## Key Features

### 1. **Multi-Modal Health Analysis**
   - PDF clinical report parsing with automated vital sign extraction
   - Direct vital sign input via web interface
   - Support for medical history factors (heart disease, smoking status)

### 2. **Advanced Machine Learning**
   - Pre-trained Artificial Neural Network (ANN) for diabetes prediction
   - 4-class health condition classification (Normal, Cardiovascular Risk, Respiratory Issue, Fever/Infection)
   - Real-time model inference with confidence scores

### 3. **AI-Powered Recommendations**
   - Integration with Ollama LLM for context-aware health recommendations
   - Streaming response generation for real-time user feedback
   - Personalized advice based on patient profile and predictions

### 4. **Clinical Data Extraction**
   - Automated regex-based text extraction from PDF reports
   - Parses: Age, BMI, Blood Pressure, Glucose levels, HbA1c, Specimen type
   - Gender and medical history detection

### 5. **Malaysian Clinical Guidelines**
   - Hypertension benchmarks (mmHg) with color-coded risk levels
   - BMI classification system (Underweight, Normal, Pre-obese, Obese classes)
   - Glucose level interpretation (Fasting & Random ranges)
   - HbA1c diabetes risk assessment

### 6. **Interactive Dashboard**
   - Real-time loading indicators
   - Color-coded health status visualization (Green/Orange/Red)
   - Guideline reference tables (collapsible)
   - Responsive design with modern UI

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend Framework** | Flask | 3.x |
| **Machine Learning** | TensorFlow/Keras | Latest |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Latest |
| **PDF Processing** | pdfplumber | Latest |
| **LLM Integration** | Ollama | Latest |
| **Preprocessing** | scikit-learn (StandardScaler, OneHotEncoder) | Latest |
| **Frontend** | HTML5, CSS3, JavaScript | - |
| **Model Storage** | joblib, HDF5 | - |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│              (Web Browser - HTML/CSS/JS)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐          ┌──────────────────┐   │
│  │  PDF Upload      │          │  Direct Input    │   │
│  │  Form            │          │  Form            │   │
│  └────────┬─────────┘          └────────┬─────────┘   │
│           │                             │              │
└───────────┼─────────────────────────────┼──────────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   Flask Backend (app.py)    │
            │   or (integrate.py/test2.py)│
            └──────────────┬──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
   │  PDF     │    │  Pre-trained │   │   Ollama    │
   │ Parsing  │    │  ANN Model   │   │    LLM      │
   │(pdfplumber)  │ (diabetes_    │   │  API        │
   │  + Regex │    │ ann_model.h5)│   │ (Llama3)    │
   └────┬─────┘    └──────┬──────┘   └──────┬──────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   Response Generation      │
            │  (Predictions + Confidence)│
            │  (Recommendations)         │
            │  (Health Assessment)       │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │  JSON Response to Frontend  │
            └────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   Display Results           │
            │  (Charts, Tables, Status)   │
            └────────────────────────────┘
```

---

## Project Structure

```
MJIC-AI-Health-Monitoring-Solution/
│
├── app.py                          # Main Flask application with PDF analysis
├── integrate.py                    # Enhanced integration module
├── test2.py                        # Extended test/development version
├── main.py                         # Basic health monitoring kiosk (direct input)
├── checkpreprocessor.py            # Utility to inspect preprocessor configuration
│
├── templates/
│   └── index.html                  # Web UI template for form and results
│
├── static/
│   └── mycaring_logo.png           # Application logo
│
├── uploads/                        # Directory for uploaded PDF files
│
├── models/
│   ├── diabetes_ann_model.h5       # Pre-trained TensorFlow/Keras ANN model
│   └── preprocessor.joblib         # Feature preprocessor (StandardScaler, OneHotEncoder)
│
├── datasets/
│   ├── diabetes_prediction_dataset.csv    # Original diabetes dataset
│   └── health_multiclass_dataset.csv      # Multi-class health dataset
│
├── Diabetes_Prediction.ipynb       # Model training & exploration notebook
│
└── README.md                       # Original project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Ollama (for AI recommendations) - [Install from ollama.ai](https://ollama.ai)
- Modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/itznotpk/MJIC-AI-Health-Monitoring-Solution.git
cd MJIC-AI-Health-Monitoring-Solution
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn tensorflow pdfplumber joblib ollama requests
```

Or using requirements file (if available):

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Ollama

1. Download and install [Ollama](https://ollama.ai)
2. Pull the Llama3 model:
   ```bash
   ollama pull llama3
   ```
3. Start Ollama service (runs on `http://localhost:11434`)

### Step 5: Verify Model Files

Ensure the following files exist in the project root:
- `diabetes_ann_model.h5` - Pre-trained model
- `preprocessor.joblib` - Feature preprocessor

---

## Running the Application

### Option 1: Diabetes Risk Assessment (PDF Analysis)

```bash
python app.py
# or
python integrate.py
# or
python test2.py
```

Then open your browser to: `http://localhost:5000`

**Usage:**
1. Upload a clinical PDF report
2. Select heart disease status (Yes/No)
3. Select smoking history
4. System extracts vital signs and predicts diabetes risk
5. Displays results with AI recommendations

### Option 2: Direct Health Monitoring Kiosk

```bash
python main.py
```

Then open your browser to: `http://localhost:5000`

**Usage:**
1. Enter vital signs:
   - Age (years)
   - Heart Rate (bpm)
   - SpO₂ (%)
   - Blood Pressure (mmHg)
   - Body Temperature (°C)
2. Click "Check Health"
3. Receive instant health assessment and AI recommendations

---

## API Endpoints

### Main Routes

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|-----------|
| GET | `/` | Load main interface | - |
| POST | `/` | Submit health data/PDF | `file` (PDF), `heart_disease`, `smoking_history` |
| POST | `/stream_recommendation` | Stream AI recommendation | - |

### Response Format

```json
{
  "result": "Low Risk / High Risk / Moderate Risk",
  "confidence": 95.2,
  "prediction_class": "Normal",
  "diagnosis": {
    "age": {"value": 45, "unit": "Years", "status": "", "color": ""},
    "bmi": {"value": 25.3, "unit": "kg/m²", "status": "Pre-obese", "color": "orange"},
    "hypertension": {"value": "130/85", "unit": "mmHg", "status": "Elevated", "color": "orange"},
    "glucose": {"value": 145, "unit": "mg/dL", "status": "Fasting High", "color": "orange"},
    "hba1c": {"value": 6.5, "unit": "%", "status": "Prediabetes", "color": "orange"}
  },
  "recommendation": "AI-generated health recommendation text..."
}
```

---

## Core Modules

### 1. **app.py / integrate.py / test2.py**

**Purpose:** Main Flask application for health assessment

**Key Functions:**
- `stream_from_ollama(prompt, model)`: Streams LLM responses
- `allowed_file(filename)`: Validates PDF uploads
- PDF text extraction using `pdfplumber`
- Regex-based vital sign extraction
- Model inference and prediction

**Features:**
- PDF file upload with validation
- Automated clinical data extraction
- Integration with pre-trained diabetes model
- Ollama LLM streaming for recommendations
- Color-coded health status visualization
- Medical guideline integration

### 2. **main.py**

**Purpose:** Simplified health monitoring kiosk

**Key Functions:**
- Direct vital sign form input
- Multi-class health prediction (Normal/Cardiovascular/Respiratory/Fever)
- Real-time AI recommendations
- Interactive result display

---

## Dataset Requirements

### For Diabetes Prediction

**Expected CSV structure** (`health_multiclass_dataset.csv`):

```csv
Age,HeartRate,SpO2,BloodPressure,Temperature,Label
45,72,98,120,37.0,0
62,85,95,135,37.2,1
...
```

**Columns:**
| Column | Type | Range | Unit |
|--------|------|-------|------|
| `Age` | int | 18-100 | Years |
| `HeartRate` | int | 50-150 | bpm |
| `SpO2` | float | 85-100 | % |
| `BloodPressure` | int | 80-200 | mmHg |
| `Temperature` | float | 35-42 | °C |
| `Label` | int | 0-3 | Class |

**Classes:**
- `0`: Normal
- `1`: Cardiovascular Risk
- `2`: Respiratory Issue
- `3`: Fever/Infection

---

## Model Details

### Neural Network Architecture

```
Input Layer (4 neurons)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(32, ReLU)
    ↓
Dense(4, Softmax) - Output Layer
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Early Stopping** | Yes (patience=5) |
| **Test Split** | 20% |
| **Preprocessing** | StandardScaler normalization |

### Model Performance

- **Input Features:** 4 (HeartRate, SpO2, BloodPressure, Temperature)
- **Output Classes:** 4 (multiclass classification)
- **Confidence Score:** Softmax probability (0-100%)

---

## Usage Examples

### Example 1: PDF Clinical Report Analysis

1. Prepare a PDF with clinical data containing:
   ```
   Age: 45
   BMI: 25.3
   Blood Pressure: 130/85
   Glucose: 145 mg/dL (Fasting)
   HbA1c: 6.5%
   ```

2. Upload via web interface
3. Select heart disease and smoking status
4. System extracts data and predicts diabetes risk
5. Displays assessment with Malaysian clinical guidelines

### Example 2: Direct Vital Sign Entry

```
Age: 35 years
Heart Rate: 78 bpm
SpO₂: 97%
Blood Pressure: 118 mmHg
Temperature: 37.0°C
```

**Expected Output:**
```
Prediction: Normal
Confidence: 94.5%
Status: All vitals within normal ranges
Recommendation: Maintain current healthy lifestyle...
```

---

## AI Integration

### Ollama LLM Setup

The system integrates Ollama (running Llama3) for generating personalized health recommendations.

**Prompt Structure:**

```
Patient Information:
- Age: 45
- Heart Rate: 78 bpm
- SpO₂: 97%
- Blood Pressure: 118 mmHg
- Temperature: 37.0°C

AI Health Model Prediction: Normal (Confidence: 94.5%)

Based on this, give short and practical recommendations:
- Immediate advice
- Lifestyle suggestion
- Whether medical attention is needed
```

**Response Streaming:**
- Real-time token streaming to frontend
- Live updates as LLM generates response
- Non-blocking async response handling

---

## Clinical Guidelines Integrated

### Hypertension Benchmarks (mmHg)
| Status | Systolic | Diastolic | Color |
|--------|----------|-----------|-------|
| Normal | <120 | <80 | Green |
| Elevated/Prehypertension | 120-139 | 80-89 | Orange |
| Hypertension Stage 1 | 140-159 | 90-99 | Red |
| Hypertension Stage 2 | ≥160 | ≥100 | Red |

### BMI Classification (kg/m²)
| Status | Range | Color |
|--------|-------|-------|
| Underweight | <18.5 | Orange |
| Normal | 18.5-24.9 | Green |
| Pre-obese | 25.0-29.9 | Orange |
| Obese Class I | 30.0-34.9 | Red |
| Obese Class II | 35.0-39.9 | Red |
| Obese Class III | ≥40.0 | Red |

### Glucose Levels (mg/dL)
| Status | Fasting | Random |
|--------|---------|--------|
| Normal | <100 | <140 |
| Prediabetes | 100-125 | 140-199 |
| Diabetes | ≥126 | ≥200 |

---

## Future Enhancements

- [ ] Real-time IoT device integration (wearables, smart health devices)
- [ ] Extended model support (cardiovascular risk, respiratory assessment)
- [ ] SHAP model explainability for detailed prediction insights
- [ ] Temporal data analysis (vital sign trends over time)
- [ ] Multi-language support
- [ ] Mobile app version (React Native/Flutter)
- [ ] Database integration for patient record management
- [ ] Advanced visualization dashboards (D3.js, Plotly)
- [ ] REST API for third-party integration
- [ ] Export reports (PDF/CSV)
- [ ] User authentication and secure data handling
- [ ] Compliance with healthcare regulations (HIPAA, etc.)

---

## Disclaimer

⚠️ **Important:** This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper medical evaluation and care.

---

## License

Project developed as part of MJIC (Malaysia Health Monitoring Initiative)

---

## Support

For issues, questions, or contributions, please refer to the project repository.
