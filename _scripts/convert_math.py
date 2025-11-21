import os
import re
import sys

def process_file(filepath):
    # print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Function to replace $...$ with $$...$$
    # But ONLY if it's not already $$...$$
    # And ignore \$ escaped dollars.
    
    # We iterate through the string to handle this safely.
    new_content = []
    i = 0
    n = len(content)
    
    while i < n:
        # Skip code blocks ``` ... ```
        if content.startswith('```', i):
            end = content.find('```', i+3)
            if end == -1: end = n
            else: end += 3
            new_content.append(content[i:end])
            i = end
            continue
            
        # Skip inline code ` ... `
        if content[i] == '`':
            j = i + 1
            while j < n and content[j] == '`': j += 1
            ticks = content[i:j]
            end = content.find(ticks, j)
            if end == -1: end = n
            else: end += len(ticks)
            new_content.append(content[i:end])
            i = end
            continue
            
        # Check for Block Math $$ ... $$
        if content.startswith('$$', i):
            end = content.find('$$', i+2)
            if end == -1: end = n
            else: end += 2
            new_content.append(content[i:end])
            i = end
            continue
            
        # Check for Inline Math $ ... $
        if content[i] == '$':
            # Check if it's escaped \$
            if i > 0 and content[i-1] == '\\':
                new_content.append('$')
                i += 1
                continue
                
            # Find closing $
            j = i + 1
            while j < n:
                if content[j] == '$':
                    if content[j-1] != '\\':
                        break
                elif content[j] == '\n':
                    # Don't span newlines for inline math usually
                    # But some people do. Let's assume no for safety.
                    j = -1
                    break
                j += 1
            
            if j != -1 and j < n:
                # Found inline math $...$
                math_content = content[i+1:j]
                # Convert to $$ ... $$ for kramdown
                new_content.append(f"$${math_content}$$")
                i = j + 1
                continue
        
        new_content.append(content[i])
        i += 1

    result = "".join(new_content)
    
    if result != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result)
        # print(f"Converted math in {filepath}")

def main():
    # Walk through all .md files in the current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        # Skip .git and _site
        if '.git' in dirs: dirs.remove('.git')
        if '_site' in dirs: dirs.remove('_site')
        
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                process_file(filepath)

if __name__ == "__main__":
    main()
