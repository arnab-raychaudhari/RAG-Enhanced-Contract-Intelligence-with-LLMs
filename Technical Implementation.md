## Purpose of this document
To provide a step by step guidance on how the solution was developed to address the
problem statement. After developing a decent comprehension, the stakeholders should
be able to make code changes to implement new features or enhance the existing
features of the application.

## Who should read this document
Intended for those interested in gaining an appreciation of the application development work or reproducing the system in a mirror environment.

## Computing Environment
The application was developed on a virtual machine hosted on Google Cloud Platform
(GCP). During development, the virtual machine was configured with 8 CPU cores and
31.35 GB of RAM. The project team did not have access to a high-performance
computing environment typically used for modern RAG LLM applications, which often
require NVIDIA A100, H100, or V100 GPUs. These GPUs are widely utilized in AI due
to their high memory capacity, multi-instance support, and tensor cores optimized for
deep learning workloads. For future scaling of this application to handle more
computing-intensive use cases, an upgrade of the computing environment is
recommended.

## Code Repository
The codebase was initially deployed on the client’s virtual machine hosted on Google
Cloud and later migrated to AWS EC2. The structure described in the Structure of Code
Repository section reflects exactly how the code was deployed to the client’s machine.

## Structure of Code Repository
The code base is organized through a directory structure that enables effortless
identification of the components on interest. The following figure depicts the structure as
it was implemented by the end of project.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/5cca064bd50041195e99e99d4c845be744f20e0f/Structure%20of%20Code%20Repository.png" width="400" />

## Before Chunking
The directory is used to download unstructured text from online sources, such as API
endpoints and PDFs hosted on various web applications. If additional sources are used
in the future, this same directory can be utilized to save those resources.

## Embedding Downloads CSV
The application provides the capability to download text embeddings once they are
created or updated in the vector database. This directory is designated for saving these
embeddings for future reference or troubleshooting purposes.

## Gephi_Images
After importing the .gdf files into Gephi and plotting network diagrams, you may need to
take screenshots or download images of these diagrams for cataloging, future
reference, and demonstrations. This directory is designated for saving such images.
Please note that there is no automated code to capture or save Gephi images in this
directory; this task must be performed manually.

## Gephi Imports GDF
A Python program has been implemented to process the text embeddings and convert
them into a .gdf file, which is essential for conducting network analysis in Gephi. This
designated directory is used to store the .gdf files produced by the Python program.

## Gephi_Projects
Throughout this project, the team created network diagrams for the application domains
studied—USSP , GAO, and Farm Bill. Each domain was analyzed separately, and
corresponding network diagrams were generated. After the diagrams are produced,
Gephi requires the resources to be saved as project files, which are stored in the
designated folder. Please note that there is no code to automate the download and
saving of these files; this task must be done manually.

## Logos
Certain logos were used on the application’s user interface, and these logos are stored
in the designated directory. In the future, if new application domains are added to the
tool, logos for these domains should also be stored in this directory.

## Notebooks
The project team implemented four Jupyter notebooks to host the application code. Two
notebooks were developed to fetch raw text from external sources, one to build the core
logic of the application, and the last one to process embeddings and create .gdf files. A
brief description of each notebook will be provided later in this document.

## Transposed Embeddings CSV
Before creating the .gdf file, an intermediate task involves building a data frame with
document chunks as column headers and listing the embeddings vertically beneath
them. The program that generates the .gdf file requires a .csv file constructed from this
data frame. These .csv files are eventually downloaded and stored in the designated
directory.

## Vector DB Embeddings
The core program for this application is designed to implement persistence when
embeddings are created and written to the vector database. This persistent logic
enhances efficiency by enabling direct retrieval of embeddings whenever needed to
answer user prompts. The persistent directories for each application domain are stored
in the designated directory.

## Notebooks - Detailed Descriptions
### Extract Text from API End Points
The notebook is titled <i>Extract_Text_from_API_EndPoint.ipynb</i> and designed to serve
the following purpose:

● Retrieve a set of PDFs from an API endpoint.

● Optionally subset the list of PDFs if only certain documents are needed.

● Extract text from the selected PDFs and save the output to a text file.

### Extract Text from Online PDFs
Use the notebook <i>Extract_Text_from_Online_PDF .ipynb</i> to perform the following steps
in sequence:

● Download a PDF from the web.

● Extract text from downloaded PDF and save the content to a text file.

● Manual download and text creation if the core process results in an error.

### Transposed Doc Embedding to GDF
The purpose of this program is to convert transposed document labels, along with their
embeddings, into a .gdf (General Data Format) file. The .gdf file will serve as the input
for creating a network analysis document using Gephi software. More details -
<a href="https://github.com/jphall663/corr_graph/blob/master/csv2gdf.ipynb" target="_blank">https://github.com/jphall663/corr_graph/blob/master/csv2gdf.ipynb</a>

### Operation Parameters
● app_prefix = set it to one of the following valid values— USSP , USSP_Light, GAO, Farm_Bill—as per your requirement.

● Transposed_dir = to read more about this directory go to the section - [Transposed Embeddings CSV](##transposed-embeddings-csv). Review and ensure you are using the correct directory.

● Gephi_import_dir = to read more about this directory go to the section - [Gephi Imports GDF](##gephi-imports-gdf). Review and ensure you are using the correct directory.

## Build RAG LLM application
The notebook titled Notebook_RAG_LLM_application.ipynb contains the core logic for
the RAG LLM application. While comments are included throughout to explain the
thought process behind each code cell, this section provides a summary to enhance
understanding.

### Main Features (Batch Mode)
#### Program Execution Time Tracking
One key consideration in developing a RAG LLM application is evaluating the time
required to generate a response to user prompts. Tracking performance becomes
especially important when high-performance computing resources are limited. T o
address this, the development team implemented a feature to capture the start and end
times of the program in batch mode, where the application can still respond to user
prompts. In batch execution, the notebook is run by pressing the “Run All” button in VS
Code, the preferred integrated development environment. The program tracks
processing time not only for the entire execution but also at key stages, such as
document chunking, embedding creation, embedding download, identifying the most
similar contextual document, and generating the LLM response. For prompts submitted
through the online channel, the Gradio interface provides feedback on processing status
in the user interface.

#### Library and Package Imports
The following libraries were imported for the purposes explained below.

● OS: A standard library providing the function to interact with the operating
system. This library was used to track and calculate the process and overall
program execution time.

● T empfile: this library was used to for creating a temporary file when the charting
functionality doesn’t have access to dimensional information to produce the 2D
charts. When charting error needs to be rendered on the user interface, a temp
file is created and returned to the online process.

● RE: a standard library used for text processing with regular expressions. This
library was used to parse the LLM response and extract data necessary for the
charting functionality.

● Glob: This library was imported to file pattern (*.txt) matching in directory listing.

● ThreadPoolExecutor: This class was implemented for concurrent execution using
threads while reading text files. This class was implemented to optimize
performance.

● Pandas: was implemented to create data frames particularly in creating .csv files
for the download of embeddings and transposed embeddings.

● Numpy: was used for numerical operations, and creating arrays, dictionaries etc.

● Word_T okenize: This function from nltk library was implemented to identify one
central word for each document chunk, this is eventually required in building the
transposed .csv file for embedding generation.

● Stopwords: this function allows the application to exclude more frequently used
stop words in text processing. Again, this was required to build the transposed
.csv file.

● Counter: this class in the Collections library is used to iterate through the
document chunk and count the number of times a certain word is used in the
text.

● String: this library was used in string manipulation while identifying the central
word in document chunk.

● TQDM: function in TQDM library was used to render progress bars while creating
the embeddings and chunking documents. Allows operations to view any monitor
the progress of operation.

● ChromaDB: ChromaDB is the vector store we implemented in this application.
Embeddings generated after chunking texts are store in ChromaDB with
persistence. Hence the chromadb package was used to define, initiate and
invoke vector stores.

● Ollama: this class from langchain_community library allows us to define LLMs
when we have installed Ollama application locally, running LLMs in local
computing environments and refraining from external API calls. Also, running
LLMs locally with Ollama keeps sensitive information in local computing
environment, which is particularly useful for apps handling confidential
information.

● RecursiveCharacterT extSplitter: this class from langchain_community was used
for splitting text.

● From_TikT oken_Encoder: this method was implemented to match the encoding
style of large language models, ensuring that the chunks fit within the model’s
token limit with certain amount of character overlaps.

● SentenceTransformer: we used this class in the sentence transformer library for
creating transformer based embeddings. Sentence Transformers are designed to
capture the semantic meaning of sentences or longer text, rather than just
individual words. Sentence Transformers offer access to a wide variety of
pretrained models, such as all-MiniLM-L6-v2, paraphrase-MiniLM, and more, which have been fine-tuned for specific tasks like semantic similarity, paraphrase
identification, and sentence classification.

● Create Retrieval Chain: This function from langchain is implemented because we
had to build a retrieval-based chain for LLMs.

● Combine Documents: We implemented this function because we had to combine
the semantically similar documents for the user query with the retrieval chain.

● Gradio: Gradio was chosen for building the user interface as the tools is well
recognized for building prototyping application with machine learning and AI use
cases.

● Contextual Compression Retriever: Since we implemented the Reranking
feature, we had to use this langchain class for retrieving contextually compressed
data.

● Cross Encoder ReRanker: This langchain class was implemented since we had
re rank documents with cross encoder.

● HuggingFaceCrossEncoder: we used this langchain class because we invoked
the BAAI/bge-reranker-v2-m3 reranking model hosted on HuggingFace.

● Matplotlib: A library for plotting and data visualization, particularly useful for
creating static, animated, and interactive graphs.

#### Choice between Llama 3.1 and Llama 3.2
Llama stands for Large Language Model meta AI, and 3.1/3.2 are version of LLM
models released by Meta (formerly Facebook) for natural language understanding and
generative AI tasks. The project team has implemented the flexibility for the Operations
team at FI Consulting and even for the end users of the RAG LLM application to select
their preferred LLM choice while prompting the application.

#### Application Domains
Four distinct application domains have been implemented, each functioning
independently with no overlaps or interactions. Users can select their desired domain
from the UI before submitting their prompts. The embeddings for each domain are
isolated, ensuring that responses generated by the application are specific to the
selected domain and not influenced by others.

● USSP: The embeddings for this domain were created using 17 PDFs from the
U.S. Department of Agriculture endpoint. Users interested in exploring the
various laws that inform contract awarding by the USDA should use this domain.

● USSP_Light: The primary difference between USSP and USSP_Light is that
instead of 17 PDFs, a single PDF was used to build a separate set of
embeddings. This domain is ideal for users with a focused interest in how the
Infrastructure Investment and Jobs Act influences contract awarding.

● GAO: This domain focuses on the Government Accountability Office (GAO). It is
suited for users seeking insights into the needs of “colonias” (economically
disadvantaged communities along the U.S.-Mexico border, primarily inhabited by
Hispanic populations). This domain provides information on the economic,
infrastructure, and environmental challenges faced by colonias and evaluates the
effectiveness and limitations of assistance programs provided by the Department
of Housing and Urban Development (HUD) and the USDA.

● Farm Bill: Users interested in conducting a comparative analysis of the 2024
Farm Bill (H.R. 8467) and the Agriculture Improvement Act of 2018 (Public Law
115-334) should select this domain.

### Document Loading
Once the raw text from external sources is downloaded in the ‘Before-Chunking’
directory, the core program is written in a way to read the text files from the directory.
This process is further made efficient by using the thread pool executor which can split
the process into multiple threads and read them simultaneously.

### Chunking
Here, we initialize a RecursiveCharacterT extSplitter with specified separators and a
chunk size of 512 characters with a 50-character overlap. Using a loop, it iterates
through each document in data, splits the text into chunks, and stores them in a chunks
list. The total number of chunks is printed, along with a preview of the first few chunks to
inspect their content. Finally, it outputs the total time taken to complete the chunking
process.

We have also implemented a functionality to print document chunks, aiding the
operations team with troubleshooting when needed.

### Embedding Model
The embedding model implemented is this application is
mixedbread-ai/mxbai-embed-large-v1, which is invoked by the SentenceTransformer
class. The model has a vector size of 1024, i.e. for every single document chunk, the
context is being compressed in a vector with 1024 dimensions, and maximum token
size of 512.

### Overwrite Document Chunk Defaults
Retrieving a large number of semantically similar document chunks from the Vector DB
for a user prompt and feeding them into the retrieval chain can sometimes overload
application memory, particularly in environments with limited RAM. This may cause the
application to run out of memory, resulting in runtime errors instead of generating the
desired response, especially on platforms with limited computing power.

To address this, the project team implemented a logic to limit the default number of
document chunks retrieved and passed to the retrieval chain. A similar approach is
applied to configure the compression retriever when the reranking feature is enabled by
the end user before submitting a prompt.

### Vector DB Collection Validation
At a time, there should only be one collection each for the four application domains. It is
important to validate if the correct collection is being used for the core logic of this
application. T o start with, the program attempts to access collections stored in the vector
store (either persistent or non-persistent, based on configuration). If collections are
found, they are sorted by the timestamp metadata in descending order, prioritizing the
most recent collections. The code then prints each collection’s name and metadata and
selects the most recent collection, noting its timestamp if available. If no collections are
found or an error occurs, it outputs an appropriate message. Finally, the total time taken
to list the collections is displayed.

### Query Embedding
Before prompting the LLM to generate a response, the user query needs to be
embedded. The same embedding model described above will be used to generate the
query embeddings. The program is written in a way to prints the length of query
embedding vector and displays the first 10 values of the embedding.

### Cosine Distance and Cosine Similarity
For troubleshooting purposes, it may be necessary to replicate the logic used to identify
the most similar document chunks for a specific user prompt. This manual replication
allows for validation of the documents being selected as context for LLM prompting.

This code inspects document embeddings from a selected collection, calculates their
cosine similarity and distance with a query embedding, and identifies the most relevant
document chunk to use as context for the language model. It begins by retrieving
documents and their embeddings from the collection, checking for the presence of both.
For each document, it computes cosine similarity and distance relative to the query
embedding and stores the results in structured lists. The document with the highest
similarity (or lowest distance) is identified and reported. Additionally, the similarity and
distance lists are sorted and displayed in descending order for analysis. If no
embeddings or documents are found, or if an error occurs, an appropriate message is
displayed. The entire process is timed, and the total time taken is printed.

### Downloading Embedding Snapshot in CSV
For any unforeseen reasons, if there is a need to view a snapshot of the embeddings,
then a feature has been implemented to read a snapshot of the embedding vectors (not all floating point numbers in a vector). This functionality has been built to retrieve embeddings, documents, IDs, and metadata from a selected collection and organizes them into a structured format. It iterates through the retrieved components, creating a dictionary for each record containing the document ID, embedding, document content, and metadata. These records are compiled into a pandas Data Frame, which is then saved as a CSV file in a specified directory. Finally, a confirmation message is displayed to indicate that the data has been successfully saved.

### Transposed Embeddings for Network Analysis
T o create network diagrams in Gephi, a .gdf file must be generated and imported into
the software. T o compile the .gdf file, each document must be assigned a one-word
header, with its vector embeddings listed vertically beneath the header.
A piece of code is written to processes and organizes document embeddings along with
their labels into a transposed Data Frame for easier analysis. It begins by ensuring
required NLTK resources are downloaded, then retrieves embeddings, documents, IDs,
and metadata from the selected collection. A function, extract_best_word, is used to
determine a label for each document based on its most common meaningful word. Each
embedding is labeled with a unique identifier in the format DocID_<id>_Label_<label>
and stored in a transposed format, where embedding values are aligned across
columns for all documents. The resulting data is saved as a CSV file, and a confirmation
message indicates the successful save.

### Initiate the Retriever
The purpose of this functionality is to dynamically set the number of document chunks
retrieved from the vector database and to optionally rerank them based on user settings
or predefined thresholds. This ensures efficient and contextually relevant retrieval while
optimizing memory usage and computational overhead.

This code determines the number of document chunks to retrieve from the vector
database based on the overwrite_doc_chunks flag and the size of chunks. It then
configures a retriever to fetch the specified number of chunks and, if reranking is
enabled, identifies the top-ranked documents to feed into the RAG LLM chain. The
selected values and actions are printed for validation.

### ReRanking
The compression of semantic meaning of text into vectors may at times lead to the loss
of suppression of relevant information. The top_k vector search documents may at time
miss the relevant context and moreover, LLMs have a limitation on how much text we
can on to them which is defined by the context window. Hence, we implemented a
ReRanking model (aka cross-encoder) – BAAI/bge-reranker-v2-m3—that given a query and document pair, will output a similarity score. Rerankers, despite being slower, are
used because they offer significantly higher accuracy than embedding models. This is
due to the limitations of bi-encoders, which compress all potential meanings of a
document into a single vector, resulting in information loss and a lack of query-specific
context, as embeddings are created before the query is received. Rerankers, on the
other hand, process raw information with each user query, allowing for a more precise,
query-specific analysis with minimal information loss. However, this improved accuracy
comes at the cost of increased processing time.

### Print the Retrieved Context for the User Prompt
We built a feature to optimize document retrieval for a user prompt, either by using a
ReRanker for high relevance or a simpler retrieval approach, while also tracking and
displaying the time taken to complete the retrieval process. This allows the application
to balance accuracy and efficiency based on user settings.

This code retrieves a response for a user prompt based on whether reranking is
required. If need_reranking is set to “Yes,” a Hugging Face cross-encoder model is
initialized as a reranker to select the top-ranked documents relevant to the prompt.
These documents are compressed and retrieved with a
ContextualCompressionRetriever, and the results are displayed. If reranking is not
needed, a standard retrieval is performed with the initial retriever. Finally, the context is
displayed in the console for review and analysis, if required.

### Retrieval based Q&A Chain
Before invoking the LLM to generate the response for user prompt, we have to build a
retrieval-based Q&A chain that leverages either standard retrieval or a reranked,
compressed retriever to feed the most relevant document content into the LLM for
question answering, allowing for flexibility based on user preferences for reranking.

The code verifies that the retriever fetches relevant documents before passing them into
a retrieval-based Q&A chain for processing. It retrieves a Q&A prompt from the hub,
initializes a document chain with the language model (LLM) and Q&A prompt, and then
extracts the page content of the retrieved documents. If reranking is enabled, the
compressed retrieval context is used; otherwise, the standard retrieved context is used.
The selected documents’ contents are combined into a single string, and the total time
taken to create the retrieval chain is displayed.

### Invoke the LLM with User Prompt and Retrieved Context
Once the retrieval based Q&A chain is defined, the next task is to execute the retrieval
chain to produce a contextually enhanced response based on the user prompt. This is
where the LLM is called to produce the contextually aware response.

The code runs the retrieval chain to generate a response to the user prompt, either with
or without reranking, and then displays the output along with the time taken. It first
checks if reranking is enabled, printing “Reranking - Yes” or “Reranking - No”
accordingly, and invokes the retrieval chain with the user prompt as input. The resulting
RAG (Retriever-Augmented Generation) response is printed, showing both the user
prompt and the model’s context-rich answer.

## Online Mode - Gradio User Interface
T o launch the Gradio UI, the operations team needs to execute the Jupyter notebook
titled Notebook_RAG_LLM_application.ipynb and wait for it to complete. Upon
execution, an HTTP URL will be generated, which can be used to access the online
interface.

### Application Domain
The end user can choose from four available application domains of interest. The
program’s modular design ensures flexibility, allowing operations to seamlessly add new
domains as the product evolves in the future.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/0d7e568aff02d7c82d4f7b85732bd41eafe4f99e/Application%20Domain.png" width="800" />

### LLM Model
At the time of implementation, two large language models—Llama3.1 and
Llama3.2—are available for end users to choose from. Additional models can be
incorporated as needed in the future.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/803345e85a978cf223abf1d77689606a32147db3/LLM%20Model%20Selection.png" width="800" />

### Prompt Text Box
A text box is provided to display preset prompts, allowing users to edit these prompts or
create new ones from scratch.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/803345e85a978cf223abf1d77689606a32147db3/Prompt%20Text%20Box.png" width="800" />

### ReRanking Toggle
Online users can enable the ReRanking feature as needed. Before clicking the RagIt,
Chart it Up, or UnRag button, they must select the application domain, choose the LLM
model, and enter the text for the prompts they require a response to.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/4e5bc5e94ac590413b16384da19f4e5d8e70bfc6/ReRanking%20Toggle.png" width="800" />

The LLM response may vary significantly depending on whether reranking is applied.
See the following images for reference.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/4e5bc5e94ac590413b16384da19f4e5d8e70bfc6/ReRanked%20Response.png" width="800" />

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/4e5bc5e94ac590413b16384da19f4e5d8e70bfc6/Response%20without%20ReRanking.png" width="800" />

### RagIt
Upon clicking this button, the application generates a contextually aware RAG
response, which is displayed in either the RAG Augmented LLM Response with
ReRanking textbox (read only) or the RAG Augmented LLM Response textbox (read
only), depending on whether reranking is applied.

### Chart it Up
Charting is often beneficial alongside textual output to provide a well-rounded
understanding. Prompt emphasis and parsing logic are implemented to prompt the LLM
to generate structured data that can be parsed and plotted on a two-dimensional chart.
When the LLM produces such data and the parsing logic successfully extracts it, a chart
will be displayed in the Generated Chart section, with an option to download the chart
image. If chart generation fails, users will see an error message indicating that a chart
could not be produced.

The Chart It Up feature is designed to generate both a chart and the LLM response,
with or without reranking, according to the user’s preference.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/4e5bc5e94ac590413b16384da19f4e5d8e70bfc6/ChartItUp.png" width="400" />

### UnRag
Comparing the RAG-enhanced response with the standard (or un-augmented) response
of the LLM model can help stakeholders appreciate the added value provided by the
RAG LLM application developed by the GWU project team. T o support this, the team
implemented a functionality that can be toggled on or off through the user interface,
allowing users to easily compare responses. This feature is available as a switch, implemented on both the front end and the backend, to meet these requirements
effectively.

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/4e5bc5e94ac590413b16384da19f4e5d8e70bfc6/UnRag%20Response.png" width="400" />

### Wrap
The Wrap button is a convenient feature that allows users to hide the UnRag Response
text area if they are not interested in viewing the unragged response.

### Context specific documents retrieved from Vector DB
The read-only text area displays the semantically similar document chunks retrieved
from the vector database in response to the user prompt, which were subsequently
used to prompt the LLM.

### Operation Parameters
1. If you change the location of the application codebook
(Notebook-LLM-RAG-Contracts.ipynb) or modify the directory structure of the
application repository (Project_LLM_and_RAG_2024_GWU), you must update
the corresponding directory paths accordingly.

● before_chunk_dir =
"/home/GWU/Project_LLM_and_RAG_2024_GWU/Before-Chunking/"

● embed_download_dir =
"/home/GWU/Project_LLM_and_RAG_2024_GWU/Embedding_Downloads_
CSV/"

● logos_dir = "/home/GWU/Project_LLM_and_RAG_2024_GWU/Logos/"

● transposed_embed_dir =
"/home/GWU/Project_LLM_and_RAG_2024_GWU/Transposed_Embeddings
_CSV/"

● vector_dir =
"/home/GWU/Project_LLM_and_RAG_2024_GWU/Vector_DB_Embeddings/"

3. Before launching the online application, you need to run the
Notebook-LLM-RAG-Contracts.ipynb in VSCode. Before you press the Run All
button on you console, you may consider changing the following parameters as
per your requirement.

● chosen_model = either set it to llama3.1 or llama3.2

● application = set it to one of the following valid values— USSP , USSP_Light,
GAO, Farm_Bill

● raw_doc_directory = at the time of writing this document, the valid values are
listed in code 6 of the notebook. If you are adding new text files, you should
set the correct value to the raw_doc_directory.

● Overwrite_doc_chunks = to read the purpose of this functionality go to the section [Overwrite Document Chunk Defaults](###overwrite-document-chunk-defaults). The valid flag values are yes and no. Also review the setting in
code cell 33.

● Need_unrag_Response = to read the purpose of this functionality go to the section [UnRag](###unrag). The valid flag values are yes and no.

● Need_reranking = to read the purpose of this functionality go to the section [ReRanking](###reranking). The valid flag values are yes and no.

● User_prompts = a set of preset user prompts were implemented while
developing this application. You may add new ones in the future, but before
you launch the application, you should select one of these prompts. The
prompts for the batch mode are listed in code cell 10-14 and for the online
mode you should review the list in code cell 38.
