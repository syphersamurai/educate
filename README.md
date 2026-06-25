# Adaptive Educator

Adaptive Educator is an intelligent, personalized learning platform that integrates Machine Learning (ML) and Large Language Models (LLM) to dynamically generate tailored educational content for students. By analyzing student metrics—such as study hours, attendance, stress levels, and motivation—the platform predicts the student's learning style and academic performance. This profile is then used alongside any neurodiverse conditions or teacher notes to synthesize a customized lesson plan that meets the student's unique needs.

## Features

- **Machine Learning Profiling**: Uses a trained Random Forest model (`scikit-learn`) to predict a student's learning style (Visual, Auditory, Reading/Writing, Kinesthetic) and grade category based on their weekly study hours, attendance, stress, and motivation levels.
- **LLM-Powered Lesson Generation**: Leverages a Large Language Model to construct a highly personalized lesson in Markdown format, tailored to the predicted learning style, specific grade level, topic, and any neurodiverse conditions provided.
- **Dynamic Image Generation**: Fetches relevant educational images server-side, avoiding client-side referral blocks, to enrich the lesson visually.
- **Web Interface**: Built with Flask, providing an intuitive form for teachers or students to input parameters and instantly view the generated lesson.

## Technical Documentation

### Architecture & Tech Stack

- **Backend Web Framework**: Flask (`app.py`) handles routing, form submission, and glues the ML and LLM components together.
- **Machine Learning**: 
  - File: `ml_predictor.py`
  - Model: Random Forest Classifier (`RandomForestClassifier` from `scikit-learn`)
  - The model predicts two targets: `LearningStyle` and `FinalGrade` using a dataset of student performance (`student_performance.csv`). 
  - Models and label encoders are cached locally using `joblib` (`student_model.pkl`, `label_encoders.pkl`) to ensure fast inference during web requests.
- **Generative AI (LLM)**: 
  - File: `llm.py`
  - Uses an external LLM API (e.g., via Hugging Face Hub) to generate structured lesson content based on a rich prompt constructed from the ML model's output and user inputs.
- **Frontend**: HTML templates rendered with Jinja2 (`templates/index.html`, `templates/lesson.html`), styled with CSS (`static/`), and formatted from Markdown using `mistune`.

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/syphersamurai/educate.git
   cd educate
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your required API keys (e.g., for the LLM) and the Flask secret key. You can use `.env.example` as a template:
   ```env
   FLASK_SECRET_KEY=your_secret_key_here
   # Add your LLM API key here, e.g. HUGGINGFACE_API_KEY=...
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   The application will automatically train and save the ML model if it's the first time running (provided `student_performance.csv` is present). It will start a local development server at `http://127.0.0.1:5000`.

### Dataset

The ML model expects a dataset named `student_performance.csv` with columns such as `StudyHours`, `Attendance`, `StressLevel`, `Motivation`, `LearningStyle`, and `FinalGrade`. The training script handles missing values and categorizes text labels automatically.

## GitHub Repository Link

**Repository URL:** [https://github.com/syphersamurai/educate](https://github.com/syphersamurai/educate)
