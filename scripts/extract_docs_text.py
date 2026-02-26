
import os
import zipfile
import re
import xml.etree.ElementTree as ET

def extract_text_from_docx(file_path):
    print(f"--- Extracting from {os.path.basename(file_path)} ---")
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # Find all text nodes in w:t
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text_nodes = tree.findall(".//w:t", namespaces)
            text = [node.text for node in text_nodes if node.text]
            print("\n".join(text)[:2000] + "..." if len(text) > 2000 else "\n".join(text))
    except Exception as e:
        print(f"Error reading docx {file_path}: {e}")

def extract_text_from_pptx(file_path):
    print(f"--- Extracting from {os.path.basename(file_path)} ---")
    try:
        with zipfile.ZipFile(file_path) as z:
            # Find slides
            slides = [f for f in z.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
            slides.sort() # Sort by name (approximate order)
            
            for slide in slides:
                xml_content = z.read(slide)
                tree = ET.fromstring(xml_content)
                namespaces = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
                # Text is usually in a:t
                text_nodes = tree.findall(".//a:t", namespaces)
                text = [node.text for node in text_nodes if node.text]
                if text:
                    print(f"\n[Slide {slide}]:")
                    print("\n".join(text))
    except Exception as e:
        print(f"Error reading pptx {file_path}: {e}")

if __name__ == "__main__":
    files = [
        "Review2 - Project Template - B.Tech.docx",
        "Rubrics_review_evaluation-REVIEW_2.docx",
        "Review_PPT_4-2_2 (1).pptx"
    ]
    base_dir = str(Path(__file__).resolve().parent.parent)
    
    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            if f.endswith(".docx"):
                extract_text_from_docx(path)
            elif f.endswith(".pptx"):
                extract_text_from_pptx(path)
        else:
            print(f"File not found: {path}")
