import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''                    if page_data.get("cover_image_path") and os.path.exists(page_data["cover_image_path"]):
                        try:
                            with open(page_data["cover_image_path"], "rb") as f:'''

replacement = '''                    is_img = page_data.get("is_image", False) or page_data.get("is_cover", False)
                    img_path = __import__('os').path.join(self.cache_dir, f"img_{page_idx}.png")
                    if is_img and __import__('os').path.exists(img_path):
                        try:
                            with open(img_path, "rb") as f:'''

content = content.replace(target, replacement)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Replaced!")
