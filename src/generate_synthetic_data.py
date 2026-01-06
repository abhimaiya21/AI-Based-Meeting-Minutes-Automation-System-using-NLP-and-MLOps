# src/generate_synthetic_data.py
import pandas as pd
import numpy as np
import os

def generate_professional_dataset(num_rows=5000):
    np.random.seed(42)
    
    # 1. Protected Attribute: Department (EXPANDED)
    depts = [
        'Engineering', 
        'Sales', 
        'HR', 
        'Marketing',
        'Finance',
        'Operations',
        'Customer Support',
        'Product Management',
        'Legal',
        'R&D',
        'IT',
        'Data Science'
    ]
    
    # More realistic distribution - Engineering and Sales are larger departments
    dept_probabilities = [0.20, 0.15, 0.08, 0.10, 0.08, 0.10, 0.12, 0.05, 0.03, 0.04, 0.03, 0.02]
    dept_col = np.random.choice(depts, size=num_rows, p=dept_probabilities)
    
    # 2. Features: Gaussian Distributions for Word Counts
    # Mimics average meeting lengths (mean ~2500 words)
    word_counts = np.random.normal(loc=2500, scale=900, size=num_rows).astype(int)
    word_counts = np.clip(word_counts, 150, 12000)
    
    # 3. Keyword Extraction simulation
    # Positive/Negative counts are percentages of the total word count
    pos_counts = (word_counts * np.random.uniform(0.02, 0.06, size=num_rows)).astype(int)
    neg_counts = (word_counts * np.random.uniform(0.01, 0.04, size=num_rows)).astype(int)
    
    # 4. Target Variable: Sentiment Score (0 to 10)
    # Formula: Base score derived from keyword ratio + Gaussian noise
    base_sentiment = 5.0 + ((pos_counts - neg_counts) / (word_counts * 0.1))
    noise = np.random.normal(0, 0.8, size=num_rows)
    
    final_score = base_sentiment + noise
    
    # --- REAL WORLD INJECTED BIASES ---
    # Multiple biases to make the governance dashboard more interesting
    
    # Bias 1: Sales gets 15% higher scores (optimistic bias)
    final_score = np.where(dept_col == 'Sales', final_score * 1.15, final_score)
    
    # Bias 2: Customer Support gets 8% lower scores (negativity bias from complaints)
    final_score = np.where(dept_col == 'Customer Support', final_score * 0.92, final_score)
    
    # Bias 3: Legal gets 10% lower scores (formal/serious tone bias)
    final_score = np.where(dept_col == 'Legal', final_score * 0.90, final_score)
    
    # Bias 4: Marketing gets 5% higher scores (positive messaging bias)
    final_score = np.where(dept_col == 'Marketing', final_score * 1.05, final_score)
    
    final_score = np.clip(final_score, 0, 10)
    
    # 5. Build DataFrame
    df = pd.DataFrame({
        'meeting_id': [f"M_{i:04d}" for i in range(num_rows)],
        'department': dept_col,
        'word_count': word_counts,
        'positive_count': pos_counts,
        'negative_count': neg_counts,
        'sentiment_score': np.round(final_score, 2)
    })
    
    # Save to directory
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_data.csv', index=False)
    
    print(f"✅ Real-world CSV generated: data/training_data.csv ({num_rows} rows)")
    print(f"\n📊 Department Distribution:")
    print(df['department'].value_counts().sort_index())
    print(f"\n📈 Sentiment Score Statistics by Department:")
    print(df.groupby('department')['sentiment_score'].agg(['mean', 'std', 'min', 'max']).round(2))

if __name__ == "__main__":
    generate_professional_dataset()

   