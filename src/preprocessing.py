"""
Data preprocessing module for SentiSight
Handles text cleaning, feature engineering, and efficient data loading for large datasets
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterator, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and preprocessing operations"""
    
    # Common patterns for cleaning
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    EXTRA_WHITESPACE = re.compile(r'\s+')
    
    # Emotion/urgency keywords for feature extraction
    URGENCY_KEYWORDS = [
        'urgent', 'asap', 'immediately', 'emergency', 'critical',
        'help', 'please help', 'stuck', 'frustrated', 'angry'
    ]
    
    NEGATIVE_EMOTIONS = [
        'angry', 'furious', 'frustrated', 'disappointed', 'upset',
        'terrible', 'horrible', 'awful', 'worst', 'hate', 'disgusted'
    ]
    
    POSITIVE_EMOTIONS = [
        'happy', 'satisfied', 'pleased', 'great', 'excellent',
        'wonderful', 'amazing', 'love', 'perfect', 'awesome'
    ]
    
    @staticmethod
    def clean_text(text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_mentions: bool = True,
                   lowercase: bool = True,
                   remove_extra_whitespace: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text to clean
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            lowercase: Convert to lowercase
            remove_extra_whitespace: Normalize whitespace
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        if remove_urls:
            text = TextPreprocessor.URL_PATTERN.sub('', text)
        
        # Remove emails
        if remove_emails:
            text = TextPreprocessor.EMAIL_PATTERN.sub('', text)
        
        # Remove mentions
        if remove_mentions:
            text = TextPreprocessor.MENTION_PATTERN.sub('', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = TextPreprocessor.EXTRA_WHITESPACE.sub(' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_features(text: str) -> Dict[str, Any]:
        """
        Extract features from text for anomaly detection and classification
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'has_urgency': any(kw in text_lower for kw in TextPreprocessor.URGENCY_KEYWORDS),
            'urgency_count': sum(text_lower.count(kw) for kw in TextPreprocessor.URGENCY_KEYWORDS),
            'negative_emotion_count': sum(text_lower.count(kw) for kw in TextPreprocessor.NEGATIVE_EMOTIONS),
            'positive_emotion_count': sum(text_lower.count(kw) for kw in TextPreprocessor.POSITIVE_EMOTIONS),
            'has_exclamation': '!' in text,
            'exclamation_count': text.count('!'),
            'has_question': '?' in text,
            'question_count': text.count('?'),
            'has_caps': any(c.isupper() for c in text),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        }
        
        return features


class DataLoader:
    """Efficiently loads and processes large datasets in chunks"""
    
    def __init__(self, file_path: str, chunksize: int = 1000):
        """
        Initialize DataLoader
        
        Args:
            file_path: Path to the CSV file
            chunksize: Number of rows to process at a time
        """
        self.file_path = Path(file_path)
        self.chunksize = chunksize
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    def load_chunks(self, columns: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks to avoid memory issues
        
        Args:
            columns: Specific columns to load (None for all)
            
        Yields:
            DataFrame chunks
        """
        logger.info(f"Loading data from {self.file_path} in chunks of {self.chunksize}")
        
        try:
            for i, chunk in enumerate(pd.read_csv(
                self.file_path, 
                chunksize=self.chunksize,
                usecols=columns
            )):
                logger.debug(f"Processing chunk {i+1} ({len(chunk)} rows)")
                yield chunk
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_sample(self, n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """
        Load a random sample of the dataset
        
        Args:
            n_rows: Number of rows to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        logger.info(f"Loading {n_rows} random samples from {self.file_path}")
        
        # First, count total rows
        total_rows = sum(1 for _ in open(self.file_path)) - 1  # Subtract header
        
        # Calculate skip probability
        skip = sorted(np.random.choice(range(1, total_rows + 1), 
                                       size=total_rows - n_rows, 
                                       replace=False))
        
        # Load skipping selected rows
        df = pd.read_csv(self.file_path, skiprows=skip)
        
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset without loading it entirely
        
        Returns:
            Dictionary with dataset info
        """
        # Read just the header
        header_df = pd.read_csv(self.file_path, nrows=0)
        columns = header_df.columns.tolist()
        
        # Count rows efficiently
        row_count = sum(1 for _ in open(self.file_path)) - 1
        
        # Get file size
        file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
        
        info = {
            'file_path': str(self.file_path),
            'columns': columns,
            'num_columns': len(columns),
            'num_rows': row_count,
            'file_size_mb': round(file_size_mb, 2)
        }
        
        logger.info(f"Dataset info: {row_count} rows, {len(columns)} columns, {file_size_mb:.2f} MB")
        return info


def preprocess_chunk(chunk: pd.DataFrame, 
                     text_column: str,
                     clean_params: Optional[Dict] = None,
                     extract_features: bool = True) -> pd.DataFrame:
    """
    Preprocess a chunk of data
    
    Args:
        chunk: DataFrame chunk to process
        text_column: Name of the column containing text
        clean_params: Parameters for text cleaning
        extract_features: Whether to extract additional features
        
    Returns:
        Processed DataFrame with cleaned text and features
    """
    if clean_params is None:
        clean_params = {}
    
    # Clean text
    chunk['cleaned_text'] = chunk[text_column].apply(
        lambda x: TextPreprocessor.clean_text(x, **clean_params)
    )
    
    # Extract features if requested
    if extract_features:
        features_list = chunk[text_column].apply(TextPreprocessor.extract_features)
        features_df = pd.DataFrame(features_list.tolist())
        chunk = pd.concat([chunk, features_df], axis=1)
    
    return chunk


def preprocess_dataset(input_path: str,
                       output_path: str,
                       text_column: str,
                       chunksize: int = 1000,
                       clean_params: Optional[Dict] = None,
                       extract_features: bool = True) -> None:
    """
    Preprocess entire dataset and save to new file
    
    Args:
        input_path: Input CSV file path
        output_path: Output CSV file path
        text_column: Name of text column
        chunksize: Chunk size for processing
        clean_params: Text cleaning parameters
        extract_features: Whether to extract features
    """
    loader = DataLoader(input_path, chunksize=chunksize)
    
    logger.info(f"Starting preprocessing: {input_path} -> {output_path}")
    
    first_chunk = True
    total_processed = 0
    
    for chunk in loader.load_chunks():
        # Preprocess chunk
        processed_chunk = preprocess_chunk(
            chunk, 
            text_column=text_column,
            clean_params=clean_params,
            extract_features=extract_features
        )
        
        # Write to output file
        mode = 'w' if first_chunk else 'a'
        header = first_chunk
        
        processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
        
        total_processed += len(processed_chunk)
        first_chunk = False
        
        logger.info(f"Processed {total_processed} rows...")
    
    logger.info(f"Preprocessing complete! Total rows: {total_processed}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/twcs.csv"
    
    # Load dataset info
    loader = DataLoader(data_path)
    info = loader.get_info()
    print(f"\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load a small sample
    print("\nLoading sample...")
    sample = loader.load_sample(n_rows=5)
    print(sample.head())
