import hashlib
known = '010E047102812FC0C18890992854220E'
paths = [
    r'C:\Program Files\MetaTrader 5 IC Markets Global',
    r'C:\Program Files\MetaTrader 5 IC Markets Global' + '\\',
    r'C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe',
    'C:/Program Files/MetaTrader 5 IC Markets Global',
]
for p in paths:
    for enc in ['utf-8', 'utf-16-le', 'ascii']:
        try:
            h = hashlib.md5(p.upper().encode(enc)).hexdigest().upper()
            tag = '  <-- MATCH' if h == known else ''
            print(f"{h}  [{enc}]{tag}")
            if tag:
                print(f"  path: {p}")
        except Exception as e:
            pass
