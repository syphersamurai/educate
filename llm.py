import os
import urllib.parse
import random
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import base64

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("[WARN] HF_TOKEN is not set. Create a .env file with HF_TOKEN=your_token "
          "(see .env.example). Lesson generation will fail until this is fixed.")

# High-performance chat models that work reliably with the chat completion API
MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct"
]

client = InferenceClient(api_key=HF_TOKEN)


def generate_tailored_lesson(profile_traits: str, grade_level: str, topic: str) -> str:
    """
    Sends a structured prompt to HuggingFace using the official SDK.
    Tries models in priority order until one succeeds.
    """

    system_prompt = f"""This is for a professional school curriculum. You are NeuroLearn AI, \
an expert and empathetic educator specialising in creating accessible, engaging content \
for neurodiverse students.

A student with the following profile needs a lesson on "{topic}".
- Grade Level: {grade_level}
- Profile / Traits: {profile_traits}

Your task is to rewrite a standard lesson about "{topic}" specifically for this student's brain.

IMPORTANT RULES:
- If the profile mentions ASD: use literal, concrete language. No idioms or metaphors. \
One idea per sentence. Use numbered micro-steps.
- If the profile mentions ADHD: use very short paragraphs. Bold key terms. Add motivational \
micro-rewards. Suggest a 5-minute break after every 3 blocks.
- Never say the words: easy, simple, simply, obviously, just, or clearly.
- Never present more than 3 new concepts at once.
- Always use an encouraging, patient, and non-judgmental tone.

Structure the response EXACTLY as follows using Markdown:

# Lesson Summary
(A brief 2-3 sentence overview of what the student will learn today)

# Step-by-Step
(Break the lesson into 3 to 5 numbered micro-steps. Short, clear, sensory-friendly sentences.)

# Comprehension Check
(3 short, low-pressure questions to check understanding)

# Engagement Tips
(2-3 personalised tips for how THIS student can best learn this topic)

# If You Are Stuck
(A simpler fallback explanation of the topic using a real-world analogy)

Do not add extra sections. Be encouraging and clear.
"""

    last_error = ""

    for model_name in MODELS:
        try:
            print(f"[LLM] Requesting model: {model_name}")

            response = client.chat_completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please generate the lesson on '{topic}' now, following the structure exactly."}
                ],
                max_tokens=1024,
                temperature=0.7
            )

            generated = response.choices[0].message.content

            if generated:
                print(f"[LLM] Success with {model_name}")
                return generated.strip()

        except Exception as e:
            last_error = f"{model_name} error: {str(e)}"
            print(f"[LLM] {last_error}")
            continue

    return (
        f"**Status: AI models are currently busy or unavailable.**\n\n"
        f"Please wait 30 seconds and try again.\n\n"
        f"_Details: {last_error}_"
    )


def generate_lesson_image_b64(topic: str) -> str:
    """
    Fetches an AI-generated educational image from Pollinations.ai SERVER-SIDE.
    Returns the image as a base64 data URI so the browser never makes
    a direct request (which would trigger the referrer block).

    Returns a base64 data URI string like:
        data:image/jpeg;base64,/9j/4AAQ...
    Or None if the image could not be fetched.
    """

    base_prompt = (
        f"Educational illustration of {topic}, "
        f"professional digital art, clear diagram style, "
        f"high quality, white background, informative, sharp focus"
    )

    encoded_prompt = urllib.parse.quote(base_prompt)
    seed = random.randint(1, 999999)

    # Use image.pollinations.ai — fetched server-side so no referrer header is sent
    image_url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?model=flux&seed={seed}&width=1024&height=1024&nologo=true"
    )

    print(f"[IMAGE] Fetching from: {image_url}")

    try:
        # Send request with NO referrer header
        response = requests.get(
            image_url,
            timeout=20,
            headers={
                "User-Agent": "NeuroLearnAI/1.0",
                "Referer": ""          # explicitly blank out the referrer
            }
        )

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            image_b64 = base64.b64encode(response.content).decode("utf-8")
            content_type = response.headers.get("Content-Type", "image/jpeg")
            data_uri = f"data:{content_type};base64,{image_b64}"
            print(f"[IMAGE] Successfully fetched and encoded image for: {topic}")
            return data_uri
        else:
            print(f"[IMAGE] Failed: status {response.status_code} — {response.text[:200]}")
            return None

    except Exception as e:
        print(f"[IMAGE] Exception while fetching image: {str(e)}")
        return None