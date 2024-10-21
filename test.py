import os
import base64
from zhipuai import ZhipuAI

# Set up dataset directory
img_directory = "dataset/test"

# Initialize ZhipuAI client
client = ZhipuAI(api_key="480b89388a064ec3f7aa99450a53b102.9Sg727GvzrS4uiO3")

# Just testing one image to see the response structure
img_name = next(img for img in os.listdir(img_directory) if img.endswith(('.jpg', '.png', '.jpeg')))
img_path = os.path.join(img_directory, img_name)

# Read the image and convert it to Base64
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

# Call the model, sending the image and task description
response = client.chat.completions.create(
    model="glm-4v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base
                    }
                },
                {
                    "type": "text",
                    "text": "This is a picture of an interactable object in an XR Game. Predict the interaction category for the given image and return the category name only. The output must be one of the following categories: {trigger, grip, joystick (click), joystick, A button, touch (reachable), touch (unreachable)}. Only provide the category name without any additional text."
                }
            ]
        }
    ]
)

# Print the whole response to understand its structure
print(response)
