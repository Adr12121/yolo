import io
try:
    with open('out_test.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        with open('last_lines.txt', 'w', encoding='utf-8') as f2:
            f2.writelines(lines[-50:])
except Exception as e:
    with open('last_lines.txt', 'w', encoding='utf-8') as f2:
        f2.write(str(e))
