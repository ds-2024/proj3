from transformers import pipeline

# STEP 1: Load the NER pipeline with a custom model
classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

# STEP 2: Define a function to classify named entities in text
def classify_entities(text):
    # Use the classifier to extract named entities
    entities = classifier(text)
    return entities

# Example text for testing
text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."

# STEP 3: Call the function to classify entities in the example text
results = classify_entities(text)

# Print the results
print(results)
