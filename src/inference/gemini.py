from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the GenAI client and specify the model
MODEL_ID = "gemini-robotics-er-1.5-preview"
PROMPT = """
          Point to no more than 10 items in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load your image
with open("test_imgs/exterior_img.png", 'rb') as f:
    image_bytes = f.read()

image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        ),
        PROMPT
    ],
    config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
)

print(image_response.text)