# Define your input text here
raw_text = r"""
[That's what metastasis effectively is.](https://www.youtube.com/watch?v=Ln3WszTq0uA&t=5782s) So we have everything completely
[5782.62 > 5789.82] (Jack) backward. And if you think about chemotherapeutic drugs, every single drug is targeted at mitosis
[5789.82 > 5795.66] (Jack) at some level. So if you really understand what I'm saying, we're doing everything wrong.
"""

# Clean the lines
cleaned_lines = []
for line in raw_text.strip().splitlines():
    cleaned_line = line.lstrip("\\") + "  "
    cleaned_lines.append(cleaned_line)

# Output to a text file
output_filename = "_formatted_output.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print(f"Formatted text written to '{output_filename}'")
