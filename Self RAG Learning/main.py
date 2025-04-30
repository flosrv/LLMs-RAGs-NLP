# === Imports ===
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === 1. Charger un modèle d'embedding open-source ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # rapide & efficace

# === 2. Tes documents sources (à indexer dans FAISS) ===
docs = [
    "Les pandas sont des mammifères herbivores originaires de Chine.",
    "Python est un langage de programmation populaire pour la data science.",
    "FAISS est une librairie de Facebook pour la recherche vectorielle rapide.",
    "DeepSeek est un modèle open-source pour coder et générer du texte.",
]

# === 3. Créer les embeddings pour les documents ===
doc_embeddings = embedder.encode(docs, convert_to_numpy=True)

# === 4. Initialiser l'index FAISS ===
dimension = doc_embeddings.shape[1]  # 384 dimensions
index = faiss.IndexFlatL2(dimension)  # index simple basé sur la distance L2
index.add(doc_embeddings)  # ajoute les vecteurs à l'index

# === 5. Question de l'utilisateur ===
user_query = "C'est quoi FAISS et à quoi ça sert ?"

# === 6. Embedding de la question ===
query_embedding = embedder.encode([user_query])

# === 7. Recherche dans FAISS (top 2 résultats) ===
top_k = 2
D, I = index.search(np.array(query_embedding), top_k)

# === 8. Récupérer les documents les plus pertinents ===
retrieved_docs = [docs[i] for i in I[0]]

# === 9. Construire le prompt pour DeepSeek ===
context = "\n".join(retrieved_docs)
final_prompt = f"""Tu es un assistant expert. Voici des infos utiles :\n{context}\n\nRépond à la question suivante : {user_query}"""

# === 10. Charger DeepSeek pour la génération ===
model_name = "deepseek-ai/deepseek-coder-1.3b-base"  # open source !
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# === 11. Génération avec pipeline Hugging Face ===
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator(final_prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]

# === 12. Afficher la réponse générée ===
print("🔍 Question:", user_query)
print("\n🧠 Connaissances trouvées :")
print(context)
print("\n✍️ Réponse générée :")
print(output.split(final_prompt)[-1].strip())
