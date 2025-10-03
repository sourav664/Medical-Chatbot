from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, filter_to_minimal_docs, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv('OPENAI_API_KEY')

os.environ['PINECONE_API_KEY'] = pinecone_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key


extracted_data = load_pdf_file(data='./data')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)


embeddings = download_embeddings()

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    
    
index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)