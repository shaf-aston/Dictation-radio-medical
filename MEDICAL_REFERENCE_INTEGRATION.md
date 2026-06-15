# Medical Reference Integration Guide

> **Scope — two separate tracks, do not mix them:**
>
> 1. **Spelling correction (DONE, shipping).** The authoritative reference for
>    *correcting mis-dictated medical terms* is the curated, hand-maintained
>    **`src/resources/radiology_lexicon.txt`** — not the textbook PDFs below.
>    The fuzzy corrector snaps a typo only to a term in that lexicon, which is
>    why it must stay clean. To improve a correction, add the term there.
> 2. **Training corpus (optional / future).** The textbook PDFs described in the
>    rest of this document are raw material for *training corpora and embeddings*
>    (Whisper fine-tune augmentation, similarity retrieval) — a different system.
>
> ⚠️ **Never feed PDF-extracted text into the spelling dictionary.** OCR /
> extraction noise (hyphenation artifacts, page headers, broken tokens) would
> pollute the snap targets and bring back the over-correction the lexicon design
> exists to prevent. PDF text is for corpora/embeddings only.

## Overview

Radio Dictate now includes a curated collection of medical radiology textbooks in `data/medical_reference/` for use as training corpora. These materials are indexed for terminology extraction, clinical reasoning patterns, and report structure learning.

## Current Status

### Downloaded ✓
- **A to Z of Chest Radiology** (1.0 MB) — Misra & Planner, 2007
  - Focus: Chest pathology patterns, pneumothorax, rib fractures, trauma findings
  - Highest-relevance for Radio Dictate's specialty (chest trauma assistant)
  - Ready for text extraction and terminology indexing

### Empty/Failed (Archive.org API Issue)
The following were marked as downloaded but are 0 bytes due to archive.org Direct PDF endpoint issues:
- A to Z of Emergency Radiology
- Basic Radiology  
- Principles of Radiographic Imaging

**Manual Fix**: Download directly from:
- Emergency: https://epdf.pub/a-z-of-emergency-radiology63bae00a0db0ceee414949d5c8394af85416.html
- Basic: https://www.abebooks.com/9781841102016 (used copies ~$15-30)
- Principles: https://archive.org/details/principlesofradi0004carl (try WebRecorder archive)

### Not Located
- Radiology Fundamentals (Harjit Singh)
- Radiology Handbook (J.S. Bensler)
- A to Z of Chest Radiology alt sources (beyond Cambridge UP paywall)

## Integration Points

### 1. Terminology Extraction

> ⚠️ **Do NOT route extracted terms into the spelling corrector** (`medical_dict`
> / `radiology_lexicon.txt`) — see the scope banner at the top. PDF-extracted
> tokens are noisy; the snapping dictionary must stay hand-curated. Use the
> snippet below only to *suggest candidate terms for human review* before a
> maintainer adds the clean ones to `radiology_lexicon.txt`.

**File**: `src/dictation/postprocess/terminology.py`

```python
from pathlib import Path
import pdfplumber

def extract_radiology_terms(pdf_path):
    """Extract medical terms from reference PDFs."""
    terms = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Tokenize and filter for medical terms
            # Match against medical_dict.py patterns
            ...
    return terms

# Usage in terminology stage:
ref_pdf = Path("data/medical_reference/a_to_z_chest_radiology.pdf")
extra_terms = extract_radiology_terms(ref_pdf)
medical_dict.extend(extra_terms)  # supplement base dictionary
```

### 2. Report Structure Learning

**File**: `src/features/adaptive_learning.py`

Use PDF text patterns to improve:
- Finding presentation order (anatomy → pathology → severity)
- Differential diagnosis formatting
- Comparison language ("No change since..." vs "New...")
- Anatomy landmark naming conventions

### 3. Embeddings & Similarity Retrieval

**File**: `src/imaging/retrieval.py` (future)

```python
from sentence_transformers import SentenceTransformer
import pdfplumber

def build_radiology_embedding_index(pdf_paths):
    """Index chest radiology text for similarity search."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    docs = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if len(text) > 100:
                    docs.append({
                        'text': text,
                        'source': pdf_path.name,
                        'page': i+1
                    })
    
    embeddings = model.encode([d['text'] for d in docs])
    # Store in vector DB (e.g., FAISS, Chroma, Qdrant)
    return {'docs': docs, 'embeddings': embeddings}

# Then query at report time:
# "similar_findings = retrieve_similar(current_report, k=5)"
```

### 4. Fine-tuning Data Augmentation

**File**: `src/cloud/tasks/whisper_voice.py`

Generate synthetic training pairs by:
1. Extract text passages from reference PDFs
2. Add realistic variations (pausing, mumbling, accents)
3. Use text-to-speech to create matching audio
4. Bundle as synthetic training pairs for voice fine-tune

```python
def augment_whisper_training_with_radiology_corpus(reference_pdf):
    """Create synthetic voice training data from radiology text."""
    with pdfplumber.open(reference_pdf) as pdf:
        passages = extract_medical_passages(pdf)
    
    # Convert passages to speech (with variability)
    synthetic_audio_pairs = []
    for passage in passages:
        audio = text_to_speech(passage, voice='radiologist-style')
        synthetic_audio_pairs.append({
            'audio': audio,
            'text': passage,
            'source': 'radiology_corpus_augment'
        })
    
    return synthetic_audio_pairs
```

## Dependencies

Install optional dependencies for PDF processing:

```bash
pip install pdfplumber sentence-transformers  # For text extraction + embeddings
```

## Workflow

1. **Extract** → `data/medical_reference/*.pdf` → text via pdfplumber
2. **Clean** → normalize whitespace, remove headers/footers, de-identify
3. **Tag** → mark medical entities, anatomical terms, pathology labels
4. **Index** → embed passages, store in vector DB or FAISS
5. **Integrate** → use in terminology.py, adaptive_learning.py, retrieval.py
6. **Train** → synthetic data generation for Whisper LoRA fine-tune

## Data Flow

```
Medical Reference PDFs
    ↓
[terminology.py] extract + supplement medical_dict
    ↓
[adaptive_learning.py] report structure patterns
    ↓
[retrieval.py] embedding index for similarity search
    ↓
[cloud/tasks/whisper_voice.py] synthetic training pairs
    ↓
[cloud fine-tune] improves model accuracy on radiologist speech
```

## Privacy & Compliance

✓ All reference PDFs are openly licensed educational materials
✓ No patient data; safe for public distribution
✓ No PHI extraction needed (unlike training data from real reports)
✓ Can be committed to repo without de-identification

## Next Steps

1. **Install pdfplumber**: `pip install pdfplumber`
2. **Manually download** missing PDFs from links above into `data/medical_reference/`
3. **Run text extraction**: See `scripts/extract_medical_reference_text.py`
4. **Update CODING_STANDARDS.md** with reference corpus usage guidelines
5. **Create integration PR**: Link terminology.py + adaptive_learning.py to corpus

## File Manifest

```
data/medical_reference/
├── metadata.json                              [source registry]
├── a_to_z_chest_radiology.pdf                 [1.0 MB ✓]
├── a_to_z_emergency_radiology.pdf             [0 bytes - download needed]
├── basic_radiology.pdf                        [0 bytes - download needed]
├── principles_of_radiographic_imaging.pdf     [0 bytes - download needed]
└── [extracted text & embeddings - future]
    ├── chest_radiology_text.txt
    ├── chest_radiology_embeddings.npy
    └── term_index.json
```

---

**Last updated**: 2026-06-12  
**Maintainer**: Radio Dictate Training Team
