import random

# Constants for SVG dimensions
SVG_WIDTH = 1170
SVG_HEIGHT = 200

text_x=-15
text_y=SVG_HEIGHT
font_size=310
text_text="DNNROI"
# text
color_one='#000000'
# misses
color_two='#111111'
# hits
color_tre='#0000FF'

nrect=300

# Function to generate a random rectangle
def generate_random_rectangle():
    x = random.randint(0, SVG_WIDTH)
    y = random.randint(0, SVG_HEIGHT)
    width = random.randint(10, 30)
    height = random.randint(5, 30)
    return (x, y, width, height)

# Function to generate the SVG content
def generate_svg_content(rectangles):
    svg_content = f"<svg xmlns='http://www.w3.org/2000/svg' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'>\n"

    # Clip Path for the text outside the rectangles (red)
    svg_content += "  <!-- Clip Path for the text outside the rectangles (red) -->\n"
    svg_content += "  <clipPath id='outsideRects'>\n"
    svg_content += f"    <rect x='0' y='0' width='{SVG_WIDTH}' height='{SVG_HEIGHT}'/>\n"
    for rect in rectangles:
        svg_content += f"    <rect x='{rect[0]}' y='{rect[1]}' width='{rect[2]}' height='{rect[3]}'/>\n"
    svg_content += "  </clipPath>\n"

    # Text: red outside the rectangles
    svg_content += "  <!-- Text: red outside the rectangles -->\n"
    svg_content += f"  <text x='{text_x}' y='{text_y}' font-size='{font_size}' fill='{color_one}' clip-path='url(#outsideRects)'>{text_text}</text>\n"

    # Draw rectangles
    # for rect in rectangles:
    #     svg_content += f"  <!-- Rectangle at ({rect[0]}, {rect[1]}) -->\n"
    #     svg_content += f"  <rect x='{rect[0]}' y='{rect[1]}' width='{rect[2]}' height='{rect[3]}' fill='{color_two}'/>\n"

    # Clip Path for the text inside each rectangle (blue)
    for i, rect in enumerate(rectangles):
        svg_content += f"  <!-- Clip Path for the text inside rectangle {i+1} (blue) -->\n"
        svg_content += f"  <clipPath id='insideRect{i+1}'>\n"
        svg_content += f"    <rect x='{rect[0]}' y='{rect[1]}' width='{rect[2]}' height='{rect[3]}'/>\n"
        svg_content += "  </clipPath>\n"

        # Text: blue inside each rectangle
        svg_content += f"  <!-- Text: blue inside rectangle {i+1} -->\n"
        svg_content += f"  <text x='{text_x}' y='{text_y}' font-size='{font_size}' fill='{color_tre}' clip-path='url(#insideRect{i+1})'>{text_text}</text>\n"

    svg_content += "</svg>"
    return svg_content

# Generate random rectangles
rectangles = [generate_random_rectangle() for _ in range(nrect)]

# Generate and save SVG content to file
with open("random_rectangles.svg", "w") as f:
    f.write(generate_svg_content(rectangles))

print(f"SVG saved to: random_rectangles.svg")
