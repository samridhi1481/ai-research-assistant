"""
Named Entity Recognition (NER) for Research Papers
This is the ADDITIONAL FEATURE requested by sir
"""

from transformers import pipeline
import spacy
import re

class NERExtractor:
    def __init__(self):
        """Initialize NER models"""
        print("🔍 Loading NER models...")
        
        # Load spaCy model (faster)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load transformer model (more accurate)
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1  # CPU
        )
        
        print("✅ NER models loaded!")
    
    def extract_entities_spacy(self, text):
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    
    def extract_entities_transformers(self, text):
        """Extract entities using transformer model"""
        entities = self.ner_pipeline(text)
        result = {}
        for entity in entities:
            label = entity['entity_group']
            word = entity['word']
            if label not in result:
                result[label] = []
            result[label].append(word)
        return result
    
    def extract_paper_entities(self, text):
        """Extract research paper specific entities"""
        # Get entities using spaCy
        entities = self.extract_entities_spacy(text)
        
        # Categorize for research papers
        categorized = {
            "models": [],
            "methods": [],
            "datasets": [],
            "metrics": [],
            "organizations": [],
            "people": [],
            "technologies": []
        }
        
        # Common ML terms to detect
        ml_models = ['bert', 'gpt', 'transformer', 'cnn', 'rnn', 'lstm', 'resnet', 
                     'yolo', 'dqn', 'gan', 'vae', 't5', 'roberta', 'xlnet']
        
        ml_methods = ['deep learning', 'machine learning', 'transfer learning', 
                      'fine-tuning', 'reinforcement learning', 'supervised learning',
                      'unsupervised learning', 'semi-supervised learning']
        
        ml_datasets = ['imagenet', 'cifar', 'mnist', 'squad', 'glue', 'coco',
                       'pascal voc', 'cityscapes', 'openimages']
        
        ml_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roc',
                      'mse', 'mae', 'bler', 'wer']
        
        # spaCy entity mapping
        entity_mapping = {
            "ORG": "organizations",
            "PERSON": "people",
            "GPE": "organizations",
            "LOC": "organizations",
            "PRODUCT": "technologies"
        }
        
        # Process spaCy entities
        for ent_type, ent_list in entities.items():
            if ent_type in entity_mapping:
                category = entity_mapping[ent_type]
                for item in ent_list:
                    if item not in categorized[category]:
                        categorized[category].append(item)
        
        # Detect ML-specific terms in the text
        text_lower = text.lower()
        
        # Check for models
        for model in ml_models:
            if model in text_lower:
                if model.capitalize() not in categorized["models"]:
                    categorized["models"].append(model.capitalize())
        
        # Check for methods
        for method in ml_methods:
            if method in text_lower:
                if method.title() not in categorized["methods"]:
                    categorized["methods"].append(method.title())
        
        # Check for datasets
        for dataset in ml_datasets:
            if dataset in text_lower:
                if dataset.capitalize() not in categorized["datasets"]:
                    categorized["datasets"].append(dataset.capitalize())
        
        # Check for metrics
        for metric in ml_metrics:
            if metric in text_lower:
                categorized["metrics"].append(metric.upper())
        
        # Clean up duplicates
        for key in categorized:
            categorized[key] = list(set(categorized[key]))
        
        return categorized
    
    def get_entities_summary(self, text):
        """Get a summary of entities found"""
        entities = self.extract_paper_entities(text)
        
        summary = {
            "total": sum(len(v) for v in entities.values()),
            "categories": {k: len(v) for k, v in entities.items() if v}
        }
        
        return summary, entities