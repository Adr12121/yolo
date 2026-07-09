with open('plan_classifier.py', 'rb') as f:
    raw = f.read()

# Fix common mojibake characters to avoid print crashes
raw = raw.replace(b'\xc7\xf8\xed\xbf\xbd\'\xed\xbf\xbd', b'-')
raw = raw.replace(b'\xc7\xf8\'\xed\xbf\xbd\xed\xbf\xbd', b'-')
raw = raw.replace(b'\xc7\xf8\xed\xbf\xbd??\xed\xbf\xbd\'\xed\xbf\xbd', b'-')

# The user saw '??'?''
raw = raw.replace(b'\xc7\xbc\xc7\xbc\'\xc7\xbc\'', b'-')
raw = raw.replace(b'\xc7\xbc\xed\xbf\xbd??\xc7\xbc\'\xed\xbf\xbd', b'-')
raw = raw.replace(b'\xc7\xf8\xed\xbf\xbd\xed\xbf\xbd\xc7\xf8', b'-')
raw = raw.replace(b'\xc7\xbc\xed\xbf\xbd??\xed\xbf\xbd\'\xed\xbf\xbd\xc7\xbc\xed\xbf\xbd??\xed\xbf\xbd\'\xed\xbf\xbd', b'-')

with open('plan_classifier.py', 'wb') as f:
    f.write(raw)

print('Cleaned up mojibake characters in strings/comments.')
