import pdfplumber, re, pandas as pd
from tqdm import tqdm

PDF_PATH = "Occu Description Vol 1.pdf"
OUT_CSV = "nco2015_vol1.csv"

# Pattern: matches codes like 8153.0111 or 8153.01
CODE_RE = re.compile(r'(\d{4}\.\d{2,4})')

rows = []
with pdfplumber.open(PDF_PATH) as pdf:
    for page in tqdm(pdf.pages, desc="Processing pages"):
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        i = 0
        while i < len(lines):
            match = CODE_RE.match(lines[i])
            if match:
                code = match.group(1)
                title = lines[i]
                desc_lines = []
                i += 1
                while i < len(lines) and not CODE_RE.match(lines[i]):
                    desc_lines.append(lines[i])
                    i += 1
                description = " ".join(desc_lines)
                rows.append({
                    "code": code,
                    "title": title,
                    "description": description,
                    "text": title + " " + description
                })
            else:
                i += 1

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} entries to {OUT_CSV}")
