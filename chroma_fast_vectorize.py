'''
Game of Thrones dataset: https://www.kaggle.com/datasets/gopinath15/gameofthrones/

Install Pytorch to use GPU: https://www.sbert.net/docs/installation.html#install-pytorch-with-cuda-support
'''
import chromadb
from chromadb.utils import embedding_functions
import time
import multiprocessing as mp
import csv

def producer(filename, batch_size, queue):

    # Load sample data (a restaurant menu of items)
    with open(filename, encoding='utf8') as file:
        lines = csv.reader(file)
        next(lines) # skip column header

        id = 2 # start id=2 to match the id with the line number in the csv (skipping the row 1 column header)

        # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
        documents = []

        # Store the corresponding menu item IDs in this array.
        metadatas = []

        # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
        ids = []

        # Loop thru each line and populate the 3 arrays.
        for line in lines:

            # Construct document usings csv values
            document = f"In season \"{line[3]}\", episode \"{line[2]}\", "
            
            if len(line[1])>0:
                document += f'{line[1]} said, \"{line[0]}\"'
            else:
                document += line[0]

            documents.append(document)
            metadatas.append({"speaker": line[1], "episode": line[2], "season": line[3]})
            ids.append(str(id))

            if len(ids)>=batch_size:
                queue.put((documents, metadatas, ids))
                documents = []
                metadatas = []
                ids = []

            id+=1

        # Queue last batch
        if(len(ids)>0):
            queue.put((documents, metadatas, ids))

# Worker function to get items from the queue
def consumer(use_cuda, queue):
    # Instantiate chromadb instance. Data is stored on disk (a folder named 'my_vectordb' will be created in the same folder as this file).
    chroma_client = chromadb.PersistentClient(path="my_vectordb")

    device = 'cuda' if use_cuda else 'cpu'

    # Select the embedding model to use.
    # List of model names can be found here https://www.sbert.net/docs/pretrained_models.html
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2", device=device)

    # Create the collection, aka vector database. Or, if database already exist, then use it. Specify the model that we want to use to do the embedding.
    collection = chroma_client.get_collection(name="got", embedding_function=sentence_transformer_ef)

    while True:
        # Check for items in queue, this process blocks until queue has items to process.
        batch = queue.get()
        if batch is None:
            break
        
        # Add to collection
        collection.add(
            documents=batch[0],
            metadatas=batch[1],
            ids=batch[2]
        )

if __name__ == "__main__":

    chroma_client = chromadb.PersistentClient(path="my_vectordb")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

    # For cleaner reloading, delete and recreate collection  
    try:
        chroma_client.get_collection(name="got")
        chroma_client.delete_collection(name="got")
    except Exception as err:
        print(err)

    collection = chroma_client.create_collection(name="got", embedding_function=sentence_transformer_ef)

    # Create a shared queue
    queue = mp.Queue()

    # Create producer and consumer processes.
    producer_process = mp.Process(target=producer, args=('game-of-thrones.csv', 1000, queue,))
    consumer_process = mp.Process(target=consumer, args=(True, queue,))
    # Do not create multiple consumer processes, because ChromaDB is not multiprocess safe.

    start_time = time.time()

    # Start processes
    producer_process.start()
    consumer_process.start()

    # Wait for producer to finish producing
    producer_process.join()

    # Signal consumer to stop consuming by putting None into the queue. Need 2 None's to stop 2 consumers.    
    queue.put(None)

    # Wait for consumer to finish consuming
    consumer_process.join()

    print(f"Elapsed seconds: {time.time()-start_time:.0f} Record count: {collection.count()}")

    