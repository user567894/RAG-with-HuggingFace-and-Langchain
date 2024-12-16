from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS #for vectorstore
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# select Hugging Face data set and column
dataset_name = "rajpurkar/squad_v2"
page_content_column = "context"

# load data set
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()

# split data into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# set embedding argument settings
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# creatae vector database from chunked documents and embed 
db = FAISS.from_documents(docs, embeddings)

# set auto tokenizer model
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

model_name = "Intel/dynamic_tinybert"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding=True, truncation=True, max_length=512
)
#create Q&A pipeline for LLM
question_answerer = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt'
)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 512},
)
# create retriever obj 
retriever = db.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False
)
question = "Who is Beyonce?"
result = qa.run({"query": question})
