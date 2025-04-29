from imports import *
from setup import *

def load_documents(DATA_PATH):
    document_load = PyPDFDirectoryLoader(DATA_PATH)
    return document_load.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def get_md5hash(file_path):
    return hashlib.md5(open(file_path,'rb').read()).hexdigest()

def calculate_chunks_ids(chunks) :
    # create unique identifier based on md5 hash of file, pg nb and chunk nb
    last_page_id = None
    current_page_index = 0
    hash_file=dict()
    
    for chunk in chunks:
        source_file = chunk.metadata.get("source")
        source = os.path.basename(source_file)
        # calculate md5 of file if not exist (avoid recomputing it)
        if source not in hash_file.keys() :
            hashsize = get_md5hash(source_file)
            hash_file.update({source : hashsize})
        else :
            hashsize = hash_file[source]
            
        page = chunk.metadata.get("page")
        #print(f"---\nsource file :{source_file}\npage nb : {page}")
        
        # compute md5 of current chunk (can retrieve it later in chroma)
        hashchunk = hashlib.md5(str(chunk.page_content).encode("utf-8")).hexdigest()
        current_page_id = f"{source}:{page}"
        
        #useless current page_id
        if current_page_id == last_page_id :
            current_page_index += 1
        else :
            current_page_index = 0
        
        chunk_id = f"{current_page_id}:{hashsize[:8]}:{hashchunk[:8]}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
        #print(f"chunk_id : {chunk_id}\n")
    
    return chunks

def add_to_chroma(chunks: list[Document]) :
    # calculate Page IDs
    chunks_with_ids = calculate_chunks_ids(chunks)
    # add or update the documents
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
        relevance_score_fn=get_distance_func()
    )
    existing_items = db.get(include=[]) #ids, included by default
    existing_ids = set(existing_items["ids"])
    print(f"ℹ️ Number of existing documents in DB : {len(existing_ids)}")
    
    # add only chunks that were never added before
    new_chunks=[]   
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids :
            new_chunks.append(chunk)

    if len(new_chunks) :
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunks_ids=[chunk.metadata["id"] for chunk in new_chunks]
        if len(new_chunks) > BATCH_SIZE :
            # split into smaller batches
            batches = [new_chunks[i:i+BATCH_SIZE] for i in range(0, len(new_chunks), BATCH_SIZE)]
            batches_ids = [new_chunks_ids[i:i+BATCH_SIZE] for i in range(0, len(new_chunks_ids), BATCH_SIZE)]

            # Process each batch separately
            for i, batch in enumerate(batches) :
                db.add_documents(batch, ids=batches_ids[i])
                print(f"✅ Documents added successfully [batch {i+1} on {len(batches)}]")

        else :
            db.add_documents(new_chunks, ids=new_chunks_ids)
            print("✅ Documents added successfully!")     
        
        #db.persist() # deprecated
        
    else :
        print(f"✅ No new documents to add")

def clear_database() :

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def parse_doc_and_add_to_db():
    # Avoid Jupyter-specific arguments that aren't relevant to the script
    if 'ipykernel_launcher' in sys.argv[0]:
        sys.argv = sys.argv[:1]  # Remove unwanted arguments for Jupyter execution
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    print("ℹ️ Loading documents from 'data' folder.")
    documents = load_documents(GB_PATH)
    print("ℹ️ Splitting document(s).")
    chunks = split_documents(documents)
    print("ℹ️ Adding document(s) to chroma DB.")
    
    # Adding tqdm for progress bar during chunk addition
    for chunk in tqdm(chunks, desc="Adding chunks to DB", unit="chunk"):
        add_to_chroma(chunk)

def get_embedding_function() :
    """
    embedding function used by langchain
    """
    return OllamaEmbeddings(
        model = LLM_MODEL_EMBED
    )

def get_distance_func() :
    return lambda distance: 1.0 - distance / 2 # None





































