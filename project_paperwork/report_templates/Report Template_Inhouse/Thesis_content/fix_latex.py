import glob
import re

files = glob.glob("/home/kbs/Documents/final_project/Report Template_Inhouse/Thesis_content/chapters/*.tex")

for path in files:
    with open(path, 'r') as f:
        content = f.read()

    # Find the latex \includegraphics line and replace its optional args
    # \includegraphics[width=0.90\textwidth]{images/data_ingestion_final.png}
    # \includegraphics[...] -> \includegraphics[width=\linewidth,height=0.85\textheight,keepaspectratio]
    
    new_content = re.sub(
        r'\\includegraphics\[.*?\]\{(images/.*?)\}',
        r'\\includegraphics[width=0.95\\linewidth,height=0.85\\textheight,keepaspectratio]{\1}',
        content
    )

    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Patched: {path}")
