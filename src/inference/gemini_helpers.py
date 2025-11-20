from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import cv2
import json
from pathlib import Path

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-robotics-er-1.5-preview"

def convert_np_to_bytes(image):

    # convert numpy array to cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, image_bytes = cv2.imencode('.png', image)
    # convert back to normal coloring
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_bytes.tobytes()

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

def query_gemini(image_bytes):
    # PROMPT = """
    #         Return bounding boxes as a JSON array with labels. Never return masks orcode fencing.
    #         Find all the objects on the table.
    #         The label returned should be an identifying name for the object detected.
    #         If an object is present multiple times, name each according to their UNIQUE CHARACTERISTIC
    #         (colors, size, position, etc.)
    #         The format should be as follows:
    #         [{"box_2d": [ymin, xmin, ymax, xmax], "label": <label for the object>}]
    #         normalized to 0-1000. The values in box_2d must only be integers.
    #         """

    PROMPT = """
            Point to all the objects on the table. The label returned
            should be an identifying name for the object detected.
            The answer should follow the json format: [{"point": <point>,
            "label": <label1>}, ...]. The points are in [y, x] format
            normalized to 0-1000.
    """

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

    return json.loads(parse_json(image_response.text))

def scale_bounding_boxes(json_output, image):
    y_scale = image.shape[0] / 1000
    x_scale = image.shape[1] / 1000
    scaled_json_output = []
    for item in json_output:
        scaled_item = {
            'box_2d': [int(item['box_2d'][0] * y_scale), int(item['box_2d'][1] * x_scale), int(item['box_2d'][2] * y_scale), int(item['box_2d'][3] * x_scale)],
            'label': item['label']
        }
        scaled_json_output.append(scaled_item)
    return scaled_json_output

def scale_points(json_output, image):
    y_scale = image.shape[0] / 1000
    x_scale = image.shape[1] / 1000
    scaled_json_output = []
    for item in json_output:
        scaled_item = {
            'point': [int(item['point'][0] * y_scale), int(item['point'][1] * x_scale)],
            'label': item['label']
        }
        scaled_json_output.append(scaled_item)
    return scaled_json_output

def plot_points(image, json_output):
    annotated_image = image.copy()
    for item in json_output:
        y, x = item['point']
        label = item['label']
        cv2.circle(annotated_image, (x, y), 5, (0, 255, 255), -1)
        text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(annotated_image, (x + 8, y - text_height), (x + text_width, y), (0, 0, 0), -1)
        cv2.putText(annotated_image, label, (x + 8, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return annotated_image

def plot_bounding_boxes(image, json_output):
    annotated_image = image.copy()

    colors = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (0, 165, 255),    # orange
        (255, 192, 203),  # pink
        (128, 0, 128),    # purple
        (42, 42, 165),    # brown
        (128, 128, 128),  # gray
        (255, 255, 0),    # cyan
    ]

    for i, item in enumerate(json_output):
        color = colors[i % len(colors)]
        y1, x1, y2, x2 = item['box_2d']
        label = item['label']

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2) # draw rectangle
        # draw label with background
        text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.rectangle(annotated_image, (x1, y2 - text_height - 10), (x1 + text_width, y2), color, -1)
        cv2.putText(annotated_image, label, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    return annotated_image
