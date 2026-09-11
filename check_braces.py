with open(r'C:\Users\arun0\OneDrive\Desktop\AegisAI\aegisai\frontend\src\pages\ChatPage.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Count braces
open_braces = 0
for i, ch in enumerate(content):
    if ch == '{':
        open_braces += 1
    elif ch == '}':
        open_braces -= 1
        if open_braces < 0:
            line = content[:i].count('\n') + 1
            col = i - content.rfind('\n', 0, i)
            print(f'Negative brace count at position {i} (line {line}, col {col})')
            print(f'Context: ...{content[max(0,i-100):i+50]}...')
            break
print(f'Final brace count: {open_braces}')