import xml.etree.ElementTree as ET
import os

def parse_ami_xml(file_path):
    """
    Reads an AMI Corpus XML file and returns a clean string.
    Handles both file paths and file-like objects.
    """
    try:
        # Check if file exists and is readable
        if isinstance(file_path, str):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"XML file not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read XML file: {file_path}")
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract all words inside <w> tags
        words = []
        for w in root.findall(".//w"):
            if w.text:
                words.append(w.text.strip())
        
        if not words:
            raise ValueError("No <w> tags with text found in XML. Ensure XML structure is correct.")
        
        # Join words to form the transcript
        return " ".join(words)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format: {str(e)}")
    except Exception as e:
        raise Exception(f"Error parsing XML: {str(e)}")