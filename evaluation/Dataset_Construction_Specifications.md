# Dataset Construction Specifications

## Wikipedia-based Synthetic Datasets

### Data Collection Process

#### Crawl Details
- **Crawl Date**: 2024-01-10
- **Source**: English Wikipedia (en.wikipedia.org)
- **Method**: Wikipedia API v1.0
- **Query Seeds**: 
  - Distinct Topics: ["machine learning", "quantum physics", "ancient history", "marine biology", "astronomy"]
  - Similar Topics: ["artificial intelligence", "machine learning", "deep learning", "neural networks", "computer vision"]
  - More Similar Topics: ["machine learning", "supervised learning", "unsupervised learning", "reinforcement learning", "deep learning"]

#### Filtering Rules
1. **Document Length**: 100-5000 words
2. **Language**: English only
3. **Quality Filter**: Articles with >10 references
4. **Topic Relevance**: Manual verification of topic alignment
5. **Duplicate Removal**: Remove near-duplicate content (similarity >0.8)

#### Preprocessing Steps
1. **Text Cleaning**: Remove HTML tags, special characters
2. **Tokenization**: Sentence-level tokenization
3. **Lowercasing**: Convert to lowercase
4. **Stopword Removal**: Remove common English stopwords
5. **Lemmatization**: Apply WordNet lemmatization
6. **Frequency Threshold**: Remove words appearing <5 times

### Dataset Statistics

#### Distinct Topics Dataset
- **Total Documents**: 1,200
- **Topics**: 5 distinct categories
- **Average Document Length**: 1,500 words
- **Vocabulary Size**: 15,000 unique words
- **Example Documents**:
  - "Machine Learning in Healthcare" (2,100 words)
  - "Quantum Computing Applications" (1,800 words)
  - "Ancient Roman Architecture" (1,600 words)

#### Similar Topics Dataset  
- **Total Documents**: 1,000
- **Topics**: 5 related AI/ML categories
- **Average Document Length**: 1,200 words
- **Vocabulary Size**: 12,000 unique words
- **Example Documents**:
  - "Introduction to Neural Networks" (1,400 words)
  - "Deep Learning for Computer Vision" (1,600 words)

#### More Similar Topics Dataset
- **Total Documents**: 800
- **Topics**: 5 closely related ML subfields
- **Average Document Length**: 1,000 words
- **Vocabulary Size**: 8,000 unique words
- **Example Documents**:
  - "Supervised Learning Algorithms" (1,200 words)
  - "Unsupervised Learning Techniques" (1,100 words)

### Topic Generation Process

#### Keyword Extraction
1. **TF-IDF Method**: Extract top 20 keywords per topic
2. **Embedding Method**: Use sentence-transformers for semantic keywords
3. **Manual Curation**: Expert review and refinement
4. **Final Topics**: 10 keywords per topic, manually verified

#### Quality Control
- **Expert Review**: 3 domain experts reviewed topic quality
- **Inter-annotator Agreement**: κ = 0.85 for topic relevance
- **Consistency Check**: Verify topic coherence across documents

### Public Dataset Integration

#### 20 Newsgroups Dataset
- **Source**: Scikit-learn datasets
- **Categories**: 20 newsgroup categories
- **Documents**: 18,846 total documents
- **Preprocessing**: Same as Wikipedia datasets
- **Usage**: External validation of evaluation methods

### Data Availability

#### Code Repository
- **GitHub**: https://github.com/username/topic-evaluation
- **Scripts**: Complete data generation and preprocessing scripts
- **Documentation**: Detailed README with reproduction instructions

#### Dataset Files
- **Raw Data**: `data/raw/` (original Wikipedia articles)
- **Processed Data**: `data/processed/` (cleaned and tokenized)
- **Topic Keywords**: `data/topics/` (extracted topic keywords)
- **Embeddings**: `data/embeddings/` (precomputed embeddings)

#### Reproduction Instructions
1. Clone repository: `git clone https://github.com/username/topic-evaluation`
2. Install dependencies: `pip install -r requirements.txt`
3. Run data generation: `python scripts/generate_datasets.py`
4. Verify results: `python scripts/verify_datasets.py`
