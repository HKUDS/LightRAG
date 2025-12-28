# Convert all uppercase letters to lowercase in testin.txt and save to testout.txt

# Read the content of testin.txt
with open("testin.txt", "r", encoding="utf-8") as infile:
    content = infile.read()

# Convert to lowercase
lowercase_content = content.lower()

# Write the lowercase content to testout.txt
with open("testout.txt", "w", encoding="utf-8") as outfile:
    outfile.write(lowercase_content)

print("Conversion complete. Output written to testout.txt.")

