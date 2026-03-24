import os
import base64
import json
import fitz
from openai import OpenAI
from PIL import Image

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_schematic_to_json(pdf_path, output_json="circuit_logic.json"):
    doc = fitz.open(pdf_path)
    all_extracted_data = []

    for page_num in range(len(doc)):
        print(f"Processing Page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(5, 5))  # Slightly higher zoom (5x)

        full_img_path = f"temp_full_p{page_num}.png"
        pix.save(full_img_path)

        # Open with Pillow for tiling
        with Image.open(full_img_path) as img:
            w, h = img.size
            # Define 4 overlapping quadrants (5% overlap to catch wires on the seams)
            mid_w, mid_h = w // 2, h // 2
            overlap = int(w * 0.05)

            tiles = [
                ("Top-Left", (0, 0, mid_w + overlap, mid_h + overlap)),
                ("Top-Right", (mid_w - overlap, 0, w, mid_h + overlap)),
                ("Bottom-Left", (0, mid_h - overlap, mid_w + overlap, h)),
                ("Bottom-Right", (mid_w - overlap, mid_h - overlap, w, h)),
            ]

            page_connections = []

            for tile_name, box in tiles:
                print(f"  Scanning {tile_name}...")
                tile_path = f"temp_tile_{tile_name}.png"
                img.crop(box).save(tile_path)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a document digitizer. Record every text label and connecting line found in this section.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "EXHAUSTIVELY list every line connection in this image section. Use JSON: source_component, source_terminal, connection_type, wire_label, target_component, target_terminal.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encode_image(tile_path)}"
                                    },
                                },
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    page_connections.extend(data.get("connections", []))

                os.remove(tile_path)

        all_extracted_data.append(
            {"page": page_num + 1, "data": {"connections": page_connections}}
        )
        print(
            f"Total connections found for Page {page_num + 1}: {len(page_connections)}"
        )
        os.remove(full_img_path)

    with open(output_json, "w") as f:
        json.dump(all_extracted_data, f, indent=2)


if __name__ == "__main__":
    process_schematic_to_json("jrs/work/mod_linx/mod_linx_data/PS10115MLC2-2.pdf")
