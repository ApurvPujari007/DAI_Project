#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install openai langchain  tiktoken pypdf unstructured[local-inference] gradio watermark\n')


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install pinecone-client\n')


# In[3]:


import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Group No. 3 PGDAI March 2023" -vmp langchain,openai,pinecone,gradio')


# In[1]:


## necessary imports
import os
from dotenv import load_dotenv
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


# In[2]:


load_dotenv()


# In[3]:


#Get the API key value from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env_key = os.getenv("PINECONE_ENV")

# print(openai_api_key)
# print(pinecone_api_key)
# print(pinecone_env_key)


# In[4]:


os.environ['OPENAI_API_KEY'] =os.getenv("OPENAI_API_KEY")


# In[1]:


### [LangChain Document Loader]
### (https://python.langchain.com/en/latest/modules/indexes/document_loaders.html)


# In[5]:


#loading our document
from langchain.document_loaders import DirectoryLoader
txt_loader = DirectoryLoader(os.getcwd(), glob="**/*.txt")


# In[6]:


# Load up your text into documents
documents = txt_loader.load()


# In[7]:


print (f'You have {len(documents)} document(s) in your data')
print (f'There are {len(documents[0].page_content)} characters in your document')


# In[8]:


documents[0]


# In[29]:


documents[1];


# In[ ]:


## Split the Text from the documents to create chunks


# In[10]:


text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500) #chunk overlap seems to work better
documents = text_splitter.split_documents(documents)
print(len(documents)) ## The entire document is split into 1780 chunks


# In[11]:


documents[0] ##first chunk of our data


# In[30]:


documents[1000] ##1000th chunk of our data


# In[2]:


## Embeddings and storing it in vectorestore i.e. creating knowledge base


# In[13]:


# Turn our texts into embeddings
embeddings = OpenAIEmbeddings()


# In[ ]:


### Using pinecone for storing vectors embeddings


# In[ ]:


### Resources:
#- [Pinecone langchain doc](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html?highlight=pinecone#pinecone)
#- What is [vectorstore](https://www.pinecone.io/learn/vector-database/)
#- Get your pinecone api key and env -> https://app.pinecone.io/


# In[15]:


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


# In[16]:


# # Run this block of code when you need to create new vector embeddings for the first time

# import pinecone 

# # initialize pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#     environment=PINECONE_ENV  # next to api key in console
# )

# index_name = "langchain-index" ## Mention your pinecone index name

# vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)


# In[17]:


## Run this block of code  if you want to use existing vector embeddings, you can load it like this

import pinecone
from tqdm.autonotebook import tqdm

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # it can be found next to api key in console
)

index_name = "langchain-index" # mention your index name here
vectorstore = Pinecone.from_existing_index(index_name, embeddings) ## fetching our existing vector embeddings


# In[ ]:


## Testing our system 


# In[18]:


query = "Writs in constitution"
docs = vectorstore.similarity_search(query)


# In[19]:


len(docs) #the system found out 4 different vectors which have similarity related to entered query


# In[20]:


print(docs[0].page_content)


# In[21]:


print(docs[1].page_content)


# In[ ]:


## Now the langchain part (Chaining with Chat History) --> `Adding Memory` to our system


# In[22]:


from langchain.llms import OpenAI


# In[23]:


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), retriever)


# In[24]:


chat_history = []
query = "What are fundamental rights"
result = qa({"question": query, "chat_history": chat_history})
result["answer"]


# In[25]:


chat_history.append((query, result["answer"]))
chat_history


# In[26]:


query = "Can you tell articles related to it"
result = qa({"question": query, "chat_history": chat_history})
result["answer"]


# In[ ]:


### As seen from above query we first asked our system about `Fundamental Rights` and followed up with next query to tell related articles about the same. The system responded perfectly giving accurate answer for both the query.

### This implies `our system can remember past query` and can give answer based on past user prompts 


# In[ ]:


## Create a chatbot having memory with simple widgets 


# In[27]:


from IPython.display import display
import ipywidgets as widgets


# In[30]:


chat_history = []

def on_submit(_):
    query = input_box.value
    input_box.value = ""
    
    if query.lower() == 'exit':
        print("Thanks for the chat!")
        return
    
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    display(widgets.HTML(f'<b>User:</b> {query}'))
    display(widgets.HTML(f'<b><font color="Orange">Chatbot:</font></b> {result["answer"]}'))

print("Chat with your data. Type 'exit' to stop")

input_box = widgets.Text(placeholder='Please enter your question:')
input_box.on_submit(on_submit)

display(input_box)


# In[ ]:


## Gradio to build  [chatbot like UI](https://gradio.app/docs/#chatbot)


# In[31]:


import gradio as gr

# Create a UI Block using gr.Blocks()
with gr.Blocks() as demo:
    # Add a Chatbot component
    gr.Markdown("<h1> <center> Welcome to the Constitution-Xpert! 🇮🇳 </center>")
    gr.Markdown("<h1> <center> 📜📃📖✒️ ⚖️🧑🏽‍⚖️👩‍⚖️_ A one stop solution to all your queries related to Constitution of India_ ⚖️🧑🏽‍⚖️👩‍⚖️ ✒️📖📜📃 </center></h1>")
    gr.Markdown("<hr>")
    gr.Markdown("<h2>How can I assist you today?</h2>")

    chatbot = gr.Chatbot(label="Constitution-Xpert: 🕵 ",height=450)
                         
    # Add a Textbox component for user input
    msg = gr.Textbox(placeholder="Input your query and then hit the enter key",label="User: 🧑🏾‍💻 ",
                     show_label=True,show_copy_button=True)

    # Add a Button to clear the chat history
    clear = gr.Button(" Clear ⛔ ")

    # Function to respond to user messages
    def respond(user_message, chat_history):
        print(user_message)
        print(chat_history)
        
        # Convert chat_history to a list of tuples for better readability
        if chat_history:
            chat_history = [tuple(sublist) for sublist in chat_history]
            print(chat_history)

        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": chat_history})
        
        # Append user message and response to chat history
        chat_history.append((user_message, response["answer"]))
        print(chat_history)
        
        return "", chat_history

    # Call the submit method to bind the respond function with UI components
    msg.submit(respond, [msg, chatbot], [msg, chatbot], queue=False)

    # Bind the Clear button to do nothing when clicked
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the UI Block with debug and sharing options
demo.launch(debug=True, share=True)

