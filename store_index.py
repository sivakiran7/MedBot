from src.helper import load_pdf_file, text_split, Huggingface_embedding_model
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='data/')
text_chunks = text_split(extracted_data)
embeddings = Huggingface_embedding_model()


pc= Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical bot"

pc.create_index(
    name = index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud ="aws",
        region="us-east-1"
    )
)

# embede eacch chunk and upsert embeddings in pine cone

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding= embeddings
)