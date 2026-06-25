import os
from flask import Flask, render_template, request, flash, redirect, url_for
import mistune
from llm import generate_tailored_lesson, generate_lesson_image_b64
from ml_predictor import predict_student_profile

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-dev-key')

markdown = mistune.create_markdown()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # --- Collect form inputs ---
        study_hours  = request.form.get('study_hours')
        attendance   = request.form.get('attendance')
        stress_level = request.form.get('stress_level')
        motivation   = request.form.get('motivation')
        condition    = request.form.get('condition', 'None')
        grade_level  = request.form.get('grade')
        topic        = request.form.get('topic')
        custom_notes = request.form.get('custom_notes', '').strip()

        # --- Validate required fields ---
        if not all([study_hours, attendance, stress_level, motivation, grade_level, topic]):
            flash("Please fill out all required fields.")
            return redirect(url_for('index'))

        try:
            # --- Step 1: Local ML model predicts learning style & grade ---
            predicted_profile = predict_student_profile(
                study_hours, attendance, stress_level, motivation
            )

            # --- Step 2: Build rich profile string for LLM ---
            notes_section = f" Additional Teacher Notes: {custom_notes}" if custom_notes else ""
            full_profile_traits = (
                f"Neurodiverse Condition: {condition}. "
                f"{predicted_profile}"
                f"{notes_section}"
            )

            # --- Step 3: Generate lesson via LLM ---
            raw_lesson_markdown = generate_tailored_lesson(
                profile_traits=full_profile_traits,
                grade_level=grade_level,
                topic=topic
            )

            # --- Step 4: Fetch image server-side (avoids referrer block) ---
            # Returns a base64 data URI string, or None if fetch failed
            image_data = generate_lesson_image_b64(topic)

            # --- Step 5: Convert markdown to HTML ---
            lesson_html = markdown(raw_lesson_markdown)

            return render_template(
                'lesson.html',
                lesson_html=lesson_html,
                topic=topic,
                profile=full_profile_traits,
                grade=grade_level,
                condition=condition,
                image_data=image_data      # base64 data URI or None
            )

        except FileNotFoundError as e:
            flash(f"Dataset error: {str(e)}. Make sure student_performance.csv is in the project folder.")
            return redirect(url_for('index'))

        except Exception as e:
            flash(f"Something went wrong: {str(e)}")
            return redirect(url_for('index'))

    return render_template('index.html')


if __name__ == '__main__':
    print("=" * 60)
    print("  Adaptive Educator — starting up")
    print("=" * 60)

    try:
        from ml_predictor import _load_models, CSV_PATH, MODEL_PATH
        if not os.path.exists(MODEL_PATH) and not os.path.exists(CSV_PATH):
            print(f"[ERROR] Cannot find {os.path.basename(CSV_PATH)} or a trained model.")
            print(f"        Place student_performance.csv in: {os.path.dirname(CSV_PATH)}")
            raise SystemExit(1)
        _load_models()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to initialise ML model: {e}")
        raise SystemExit(1)

    print("=" * 60)
    print("  Ready! Open http://127.0.0.1:5000 in your browser.")
    print("=" * 60)
    app.run(debug=True, port=5000)