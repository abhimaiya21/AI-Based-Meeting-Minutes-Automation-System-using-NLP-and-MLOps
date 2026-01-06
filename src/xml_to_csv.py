# File: src/xml_to_csv.py
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os

def parse_and_process_xml():
    print("⚙️ Parsing 'EN2004a.A.words.xml' for Training...")
    
    # 1. PATH TO YOUR SPECIFIC FILE
    xml_path = "data/EN2004a.A.words.xml"
    
    if not os.path.exists(xml_path):
        print(f"❌ Error: Could not find {xml_path}")
        return

    # 2. PARSE THE XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        all_words = []
        # Find all 'w' tags (words) and ignore namespaces
        for child in root.iter():
            if child.tag.endswith('w') and child.text:
                all_words.append(child.text)
                
        print(f"   Found {len(all_words)} words in transcript.")

        # 3. CHUNK INTO SMALL "MEETINGS" (Data Augmentation)
        # We split one big meeting into many small 30-word chunks to train the model
        chunk_size = 30
        rows = []
        
        for i in range(0, len(all_words), chunk_size):
            chunk = all_words[i:i+chunk_size]
            if len(chunk) < 10: continue

            text_segment = " ".join(chunk).lower()
            
            # 4. CALCULATE SCORES (Heuristic Labeling)
            # We teach the model: "Yeah/Right" is good, "No/Sorry" is bad.
            pos_words = text_segment.count("yeah") + text_segment.count("right") + text_segment.count("okay") + text_segment.count("good")
            neg_words = text_segment.count("no") + text_segment.count("sorry") + text_segment.count("but") + text_segment.count("problem")
            
            # Target Score (0 to 10)
            score = 5.0 + (pos_words * 0.5) - (neg_words * 0.5)
            score = np.clip(score, 0, 10)

            rows.append({
                'word_count': len(chunk),
                'positive_count': pos_words,
                'negative_count': neg_words,
                'sentiment_score': round(score, 2)
            })

        # 5. SAVE CSV
        df = pd.DataFrame(rows)
        output_path = "data/training_data.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Success! Created {len(df)} training rows at '{output_path}'")
        
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")

if __name__ == "__main__":
    parse_and_process_xml()