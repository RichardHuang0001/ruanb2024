import base64
from zhipuai import ZhipuAI

# 读取图片并进行Base64编码
img_path = "/Users/huangwei/Desktop/vr图片/152139.jpg"
with open(img_path, 'rb') as img_file:
    img_base = base64.b64encode(img_file.read()).decode('utf-8')

# 初始化ZhipuAI客户端
client = ZhipuAI(api_key="480b89388a064ec3f7aa99450a53b102.9Sg727GvzrS4uiO3")  # 填写API Key

# 调用模型，发送图片和任务描述
response = client.chat.completions.create(
    model="glm-4v",  # 使用指定的模型
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
                    "text": "Based on the uploaded image and the provided related description,\
                     identify all interactive objects within the VR environment. For each interactive object, \
                     determine the interaction category by selecting **one** of the following interaction methods: \
                     {trigger, grip, joystick (click), joystick, A button, touch (reachable), touch (unreachable), look at, shoot, watering, throw, hold, select, knock, unlock, grab, shooted, hit, sweep, open, wear, daub, contain, infiltration, wipe, adsorption, drop, scan, rotate, carve, body matching, summon, wither, paint, drum, burst, collide, play, build, press, squeeze, fire, spin, drip, eat, put on, stab, pull, throw at, put head in, hit (with bat), move gamepad, hit (with ball)}.Output the result in the following format: `{object, interaction category}`. Only use the interaction categories provided above, and do not provide any additional explanation or analysis."
                }
            ]
        }
    ]
)

# 输出模型的响应
print(response.choices[0].message)

# "Based on the uploaded image and the provided related description, identify all interactive objects within the VR environment. For each interactive object, determine the interaction category by selecting **one** of the following interaction methods: {trigger, grip, joystick (click), joystick, A button, touch (reachable), touch (unreachable), look at, shoot, watering, throw, hold, select, knock, unlock, grab, shooted, hit, sweep, open, wear, daub, contain, infiltration, wipe, adsorption, drop, scan, rotate, carve, body matching, summon, wither, paint, drum, burst, collide, play, build, press, squeeze, fire, spin, drip, eat, put on, stab, pull, throw at, put head in, hit (with bat), move gamepad, hit (with ball)}.Output the result in the following format: `{object, interaction category}`. Only use the interaction categories provided above, and do not provide any additional explanation or analysis."
