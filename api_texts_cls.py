from fastapi import FastAPI, Form
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

# STEP 1: Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

# STEP 2: Initialize FastAPI app
app = FastAPI()

# STEP 3: Define the POST endpoint for text classification
@app.post("/textClassification/")
async def text_classification(text: str = Form(...)):
    # STEP 4: Prepare input texts with required prefixes
    input_texts = ['query: ' + text]  # Assuming only one text is passed per request

    # STEP 4-1: Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    # STEP 4-2: Perform inference using the model
    with torch.no_grad():
        outputs = model(**batch_dict)

    # STEP 4-3: Post-process the outputs to get embeddings
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity scores
    scores = (embeddings[0] @ embeddings[1:].T) * 100  # Assuming embeddings[0] is query, rest are passages

    # Convert scores to list and return
    scores_list = scores.tolist()
    return {"scores": scores_list}
