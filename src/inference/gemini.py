from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import cv2
import json

load_dotenv()

def parse_json(json_output):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  for i, line in enumerate(lines):
    if line == "```json":
      # Remove everything before "```json"
      json_output = "\n".join(lines[i + 1 :])
      # Remove everything after the closing "```"
      json_output = json_output.split("```")[0]
      break  # Exit the loop once "```json" is found
  return json_output

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
image = cv2.imread("test_imgs/exterior_img.png")
with open("./test_imgs/exterior_img.png", 'rb') as f:
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
output = parse_json(image_response.text)
print(output)

# create new image with the points annotated
annotated_image = image.copy()
for item in output:
  print(item)
  cv2.circle(annotated_image, (item['point'][1], item['point'][0]), 10, (0, 0, 255), 2)
  cv2.putText(annotated_image, item['label'], (item['point'][1], item['point'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.imwrite("annotated_image.png", annotated_image)