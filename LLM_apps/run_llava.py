import ollama
import glob
import cv2
from pathlib import Path

def write_image(file_path: str, title: str) -> None:

    img = cv2.imread(file_path)
    height, width, _ = img.shape

    filename = Path(file_path)
    org = (20, height - 20)
    fontScale = 0.75
    color = (255, 0, 255)
    thickness = 2
    image = cv2.putText(img, title, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(f"ts/{filename.stem}_detected_{filename.suffix}", image)

images = [f for f in glob.glob("ts/*.*", recursive = True)]

for image in images:

    filename = Path(image).name
    res = ollama.chat(
        model="llava",
        messages=[
            {
                'role': 'user',
                'content': 'Tell me what this traffic sign is. Just say the name and nothing else, do not describe. Just say the traffic sign name:',
                'images': [image]
            }
        ]
    )

    print(f"{filename} -> {res['message']['content']}")
    write_image(image, res['message']['content'])
