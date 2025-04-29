from langchain_ollama import OllamaEmbeddings


# paths for files import and db location
GB_PATH = "Gabriel_Marquez_Books" 
PERSIAN_DATA = "PERSIAN_DATA"
CHROMA_PATH = "chroma" # path for chroma DB, it'll create one if missing

# config for LLMs
LLM_MODEL = "llama3.2" 
LLM_MODEL_EMBED = "mxbai-embed-large" #"nomic-embed-text" # LLM_MODEL
LLM_TEMP = 0.2 # temperature, default 0.8
NUM_CTX = 8000 # size of context in token, default 2048
TOP_K = 5 # reduce probability to provide non sense text, lower, more conservative; default, 40

# pdf file serialization in documents (for populate step)
CHUNK_SIZE = 400 # Chunk size 
CHUNK_OVERLAP = 80
BATCH_SIZE = 5432 # Maximum batch size allowed, chroma, default 5432?

# template regarding prompts
K_RESULTS = 5 # number of answwer from chroma that will be prepend to query

# the prompt template can be modified for better results
PROMPT_TEMPLATE = """
Answer the question based strictly between <contexts> tag :
<contexts>
{context}
</contexts>
Answer the question based strictly on the above <contexts> tag .
Apply the following rules set between <rules> tag below:
<rules>
1) Choose the most consistents context regarding the question in the <question> tag.
2) Use the <source> tag content in use with your answers and related to the corresponding <context> tag content. What you say must be sourced such as a consistent bibliography.
3) If all <context> are irrelevant to the question between <question> tag, do not answer.
4) Detect question defined within the <question> tag and answer using the same language.
</rules>
Applying whole <contexts> tag content using the set of <rules> tag content, try to answer to the question below in <question> tag :
<question>
{question}
</question>
"""
PROMPT_TEMPLATE = """
<contexts>
{context}
</contexts>
<rules>
1) Choose the most consistents context regarding the question in the <question> tag.
2) Use the <source> tag content in use with your answers and related to the corresponding <context> tag content. What you say must be sourced such as a consistent bibliography.
3) If all <context> are irrelevant to the question between <question> tag, do not answer.
4) Detect question defined within the <question> tag and answer using the same language.
</rules>
<question>
{question}
</question>
Using only whole <contexts> tag content by applying the set of <rules> tag content, answer exhaustively to the question set with in <question> tag.
"""
PRINT_PROMPT = True # print all the prompt sent to LLMs model before the request. YEXITou can check what's inside.




