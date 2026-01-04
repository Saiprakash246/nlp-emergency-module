# NLP Module – Emergency Call Analysis
## Overview
This module processes transcribed emergency call text and extracts:
- Emergency type
- Urgency level
- Location information

It is designed to integrate with the AI Emergency Call Intelligence System.

## Features Implemented
- Text cleaning and normalization
- Emergency classification (Fire, Medical, Accident, Crime)
- Urgency detection (LOW / MEDIUM / HIGH / CRITICAL)
- Location extraction using spaCy NER with fallback rules
- Raw scenario testing with F1-score evaluation

## Example Input
fire accident in chittoor please help immediately


## Example Output
- Emergency Type: Fire  
- Urgency Level: HIGH  
- Location: chittoor  

## Evaluation
- F1-score calculated using raw emergency scenarios
- Controlled tests may show high accuracy
- Ambiguous cases highlight limitations of rule-based logic

## Future Enhancement
- BERT-based fallback for unseen or ambiguous emergency descriptions
  (e.g., "vehicle overturned", "collapsed building")

## Notes
- This repository is for **module review**
