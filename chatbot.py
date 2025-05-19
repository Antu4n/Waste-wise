#import the required libraries
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration    
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("punkt_tab")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import util

save_path = './waste-qa'
df = pd.read_csv(f'{save_path}' + "/augmented_df.csv")

def load_model():
    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(save_path)

    # Load the model architecture (from config)
    model = T5ForConditionalGeneration.from_pretrained(save_path)

    # Load the trained weights (from state_dict)
    model.load_state_dict(torch.load(save_path + "/waste-qa.ckpt", map_location=torch.device('cpu')))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to {device}")

    return model, tokenizer
model, tokenizer = load_model()
print(next(model.parameters()).device)
print("Model and tokenizer loaded successfully")

    
#load sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
#get question embeddings
question_embeddings = embedder.encode(df["question"].tolist(), convert_to_tensor=True)


#return the top question answer matches from the dataset.
def get_top_matches(question, top_k=3, threshold=0.6):
    # Embed user question
    user_embedding = embedder.encode(question, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = torch.nn.functional.cosine_similarity(user_embedding, question_embeddings)

    # Get top-k most similar questions
    top_results = torch.topk(similarities, k=top_k)

    matches = []
    for score, index in zip(top_results[0], top_results[1]):
        if score.item() >= threshold:
            matches.append((df.iloc[index.item()]["question"], df.iloc[index.item()]["answer"], float(score)))

    return matches


def generate_answer(question, top_k=3):
  if not get_top_matches(question):
    return "Sorry, I didn't quite get that, could you rephrase your question?"

  #use the top match for t5 generation
  matched_question, matched_answer, similarity = get_top_matches(question)[0]


  t5_input = f"question: {question} context: {matched_answer}"

  source_encoding = tokenizer(
      t5_input,
      max_length = 396,
      padding = "max_length",
      truncation = "only_second",
      return_attention_mask = True,
      add_special_tokens = True,
      return_tensors = "pt"
  )

  #move the tensors to the correct device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_ids = source_encoding["input_ids"].to(device)
  attention_mask = source_encoding["attention_mask"].to(device)

  generated_ids = model.generate(
      input_ids = input_ids,
      attention_mask = attention_mask,
      num_beams = 4,
      do_sample = False,
      # max_length = 256,
      max_new_tokens = 100,
      # min_length = 20,
      # top_k = 50, # considers only the top 50 tokens for sampling
      # top_p = 0.9, #choose the smallest set of tokens whose cumulative probability exceeds 0.9
      repetition_penalty = 2.5,
      length_penalty = 1.5,
      # early_stopping = True,
      use_cache = True
  )

  preds = [
      tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      for gen_id in generated_ids
  ]
  preds = "".join(preds)
  # return f"Based on what I know:\n{preds}\n\nMatched Q: \"{matched_question}\" (score: {similarity:.2f})"
  return preds

#evaluate the generated response using BLEU and cosine similarity
smoothie  = SmoothingFunction().method4

def evaluate_response(predicted: str, reference: str) -> dict:
    # BLEU Score
    reference_tokens = [word_tokenize(reference.lower())]
    predicted_tokens = word_tokenize(predicted.lower())
    bleu = sentence_bleu(reference_tokens, predicted_tokens, smoothing_function=smoothie)

    # Cosine Similarity
    pred_embedding = embedder.encode(predicted, convert_to_tensor=True)
    ref_embedding = embedder.encode(reference, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(pred_embedding, ref_embedding).item()

    return {
        "generated_response": predicted,
        "reference_response": reference,
        "BLEU Score": round(bleu, 4),
        "Semantic Similarity": round(cosine_sim, 4)
    }
#run the chatbot simulation
def run_chatbot(question):
    #input question
    # question = input("Ask a question: ")
    if question.lower() == "exit":
        return {
            "response": "Thank you for using the Waste Management Chatbot!",
            "reference": None,
            "bleu": None,
            "similarity": None
        }
    #generate response.
    predicted = generate_answer(question)
    #get reference from the knowledge base
    try:
        top_match = get_top_matches(question)[0]
        reference = top_match[1] #retrieve the reference answer
        #evaluate the predicted response
        results = evaluate_response(predicted, reference)
        print("\nEvaluation Metrics")
        print("-" * len("Evalution Metrics"))
        print("bleu score: ", results['BLEU Score'])
        print("similaity score: ", results['Semantic Similarity'])
        return {
            "response": predicted,
            "reference": reference,
            "bleu": results['BLEU Score'],
            "similarity": results['Semantic Similarity']
        }
    except IndexError:
         return {
            "response": "Sorry, I didn't quite get that. Could you ask something related to waste, recycling or disposal?",
            "reference": None,
            "bleu": None,
            "similarity": None
        }