# RAGov-contract-analysis
Leveraging AI to Enhance Data Driven Insights for Government Contracts

## Business Practicum Engagement
This project was undertaken, executed, and completed as part of the Master of Science in Business Analytics program at George Washington University. As a requirement of the degree, I was tasked with partnering with a real-world business (my client) to address an ongoing problem using advanced Artificial Intelligence techniques. The project spanned four months and was successfully delivered to the client in December 2024.

![](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/c41327c9dfd913888d6b5b760d95fc446e8da624/Architecture-Diagram-GIF.gif)

## Executive Summary
FI Consulting faces a significant need for a reliable system capable of accurately retrieving and
synthesizing information from complex, unstructured documents to support high stakes decision
making in fields like legislative analysis, government procurement, and consulting. Traditional
large language models (LLMs), while advanced, often produce responses that seem plausible but
lack factual grounding, limiting their utility where accuracy is crucial. This project aims to address
these limitations by developing a proof-of-concept Insight Engine that leverages Retrieval
Augmented Generation (RAG) techniques. By combining LLM capabilities with retrieval
mechanisms, this system anchors responses in relevant, retrieved content, effectively reducing
ungrounded “hallucinations” and improving response accuracy and contextual reliability.

The Insight Engine is designed to deliver precise, real-time information, making it a scalable tool
adaptable to FI Consulting’s diverse needs. Its functionality extends beyond legislative and
contract analysis, allowing it to handle a broad range of unstructured data types, from policy
documents to complex financial reports. Initial testing demonstrated qualitative improvements in
response accuracy and relevance over standard LLM outputs, supporting the system’s value in
applications that require precision and dependability. Additionally, an interactive Gradio interface
provides users with a seamless, real-time experience, enabling them to explore grounded and
ungrounded responses side by side, enhancing both usability and transparency.

Although this project does not implement formal risk mitigation measures, the report outlines key
potential risks associated with deploying RAG and LLM systems, particularly in data sensitive
environments relevant to FI Consulting. These risks include considerations around data accuracy,
bias, and privacy, all of which are critical when utilizing AI driven systems in high stakes
applications. Recognizing these risks underscores the importance of responsible and ethical
deployment practices for the Insight Engine, should FI Consulting choose to scale the system.
Responsible deployment would involve establishing frameworks for ongoing monitoring, regular
audits, and secure data management. By acknowledging these potential challenges, the report
emphasizes the need for a cautious, ethical approach to the system’s future.

## Introduction
In recent years, advances in Large Language Models (LLMs) have significantly expanded the
applications of artificial intelligence in natural language processing (NLP). Models such as
OpenAI’s GPT, Llama, and Mistral can generate detailed and contextually relevant responses,
enabling a range of applications from conversational agents to content generation. However,
despite their sophistication, LLMs have inherent limitations. While they can produce linguistically
coherent responses, they often suffer from "hallucinations", instances where the models generate
plausible sounding but factually inaccurate information (Brown et al., 2020; Ji et al., 2023). This
shortfall can be problematic, particularly in sectors where accuracy is essential such as, but not
limited to healthcare, law, and federal contract.

To overcome these limitations, Retrieval Augmented Generation (RAG) has emerged as a
powerful technique that enhances the factual accuracy and contextual relevance of LLM outputs
(Lewis et al., 2020; Hugging Face Blog, 2023). RAG supplements the generative capabilities of
LLMs by integrating an external retrieval system that pulls in relevant information from a verified
knowledge base. This approach grounds the model’s responses in real data, significantly
improving their reliability and applicability in high-stakes domains. RAG’s hybrid structure, which
combines retrieval with generation, allows for more precise, contextually aware responses making
it an asset in applications where factual grounding is crucial.

## Background
FI Consulting recognized the potential of RAG to address information retrieval challenges within
complex fields such as federal contracting and legislative analysis. FI Consulting, a data analytics
and technology firm, is deeply invested in providing reliable and efficient solutions for clients
dealing with complex data portfolios. By harnessing RAG, the firm aimed to explore new ways of
automating information retrieval and summarization, particularly within datasets requiring high
levels of accuracy and specificity.

Initially, the project centered on developing a system that could answer detailed queries related to
federal contracts. The system would ideally enable users to explore complex contract structures,
funding allocations, and inter agency agreements all areas where manual analysis is typically time
intensive and laborious. However, as the project evolved, practical constraints such as limited
access to comprehensive federal data and computational resources necessitated a refined focus.
Recognizing these limitations, the team adapted the scope to prioritize a proof-of-concept (POC)
that could showcase the potential of RAG within a more targeted use case.

The final iteration of the project focused on legislative and analysis, using the 2018 Farm Bill and
the proposed 2024 Farm Bill as the primary datasets. This approach allowed FI Consulting to
demonstrate RAG’s effectiveness in a structured, high impact context specifically, by enabling the
system to perform comparative analysis on legislative content. By summarizing key changes in
policy and funding between legislative versions, this use case illustrated how RAG could
streamline complex document analysis, allowing stakeholders to quickly and accurately track
modifications.

In this report, we document the evolution of the project from concept to POC, detailing the
technical and strategic adjustments made to accommodate practical constraints. This case study
highlights the capabilities and value of RAG in automating information retrieval and analysis,
setting the foundation for future applications within FI Consulting’s portfolio.

## Problem Statement
Organizations operating in data intensive fields, such as federal legislative and contract analysis,
face mounting challenges in efficiently retrieving, analyzing, and synthesizing information from
complex, unstructured datasets. FI Consulting often find themselves spending excessive time and
resources navigating dense legislative documents and contractual data to extract relevant insights.
This inefficiency is compounded by trust issues stemming from inconsistent and unreliable
interpretations of data, which ultimately hampers their ability to make informed decisions with
confidence.

Traditional LLMs, such as GPT and Llama, while powerful in natural language processing,
struggle with critical limitations like “hallucinations” generating responses that sound accurate but
lack factual grounding (Ji et al., 2023). These models often lose context and produce inconsistent
responses, making them unreliable tools for applications requiring high precision, such as
identifying legislative policy changes, funding adjustments, or key contractual terms. For example,
analyzing the 2018 Farm Bill and the proposed 2024 Farm Bill to track changes in policy and
funding allocations is labor intensive and error prone when done manually, and LLMs without
augmentation fail to provide the level of accuracy required for such tasks.

To address these issues, this project explores the application of RAG to enhance LLM reliability
by grounding responses in real-time retrieved data from authoritative sources. By leveraging RAG,
FI Consulting aims to develop a scalable POC system capable of handling unstructured datasets
such as, but not limited to Public Laws and legislative documents. This system focuses not only
on the Farm Bill but also on other laws integral to the client’s needs, for instance those addressing
healthcare (Public Law 115-123), disaster relief (Public Law 115-56), infrastructure investment
(Public Law 117-58), and retirement security (SECURE 2.0 Act of 2022). These datasets represent
the diverse and complex legal content the system must accurately analyze to support the client’s
decision-making processes.

The integration of RAG allows for accurate, contextually grounded responses, significantly
reducing manual effort and enabling stakeholders to gain timely insights into policy changes,
contractual obligations, and funding distributions. By addressing the inefficiencies, trust issues,
and resource burdens associated with traditional methods, this system has the potential to
revolutionize the way FI Consulting’s clients interact with unstructured data, positioning RAG as
a critical solution for their broader data-driven needs.

## Objectives
The objective of this project is to empower FI Consulting to operate more efficiently and
confidently by providing timely, accurate insights from complex, unstructured datasets. Once
deployed, the Insight Engine will enable users, such as marketing and sales teams, to quickly
access contextually relevant information, transforming how they navigate and interpret data-
intensive documents like legislative texts, contracts, and public laws. This tool aims to reduce the
manual effort and resource expenditure currently needed to retrieve critical information, allowing
teams to focus on strategic, value-driven tasks rather than labor-intensive data processing.

By anchoring AI generated responses in real-time, authoritative data through Retrieval Augmented
Generation this solution aspires to build trust in AI outputs, minimizing inaccuracies and
enhancing decision making reliability. The end goal is to foster a more agile, informed, and
productive organizational environment where teams can confidently leverage insights for client
engagements, policy analysis, and business development. Through this project, FI Consulting
seeks to establish a state where data retrieval is seamless, accurate, and actionable ultimately
supporting smarter, faster, and more effective decision-making across the organization.
The study set out to:

1. Combine LLMs with External Data Retrieval: Develop a system that grounds LLM
responses in accurate, contextually relevant data from external sources, reducing common
issues like unsubstantiated claims and misrepresentations.

3. Implement Efficient Preprocessing for Document Analysis: Create a robust pipeline to
prepare large documents for retrieval and analysis, enabling quick and relevant data access.

5. Refine Response Relevance through Reranking: Apply reranking techniques to prioritize
the most relevant information, ensuring that users receive responses that are both accurate
and contextually appropriate.

7. Create an Interactive and Scalable User Interface: Design a user-friendly interface that
allows users to interact with the RAG system and experience the enhanced accuracy it
offers.

9. Demonstrate Practical Use with a Legislative Comparison Use Case: Apply the system to
legislative documents to showcase its value in analyzing and comparing complex, data rich
content.

These objectives were intended to demonstrate RAG’s potential to improve LLM reliability,
ultimately providing a foundation for future applications in data intensive fields within FI
Consulting.

## Architecture Diagram

<a href="https://lucid.app/lucidchart/02be26cc-ce83-46de-85b4-15e97733f903/view?page=0_0#" target="_blank">RAGov: Contract Analysis</a>

## Methodology
This section outlines the comprehensive approach taken to develop a proof-of-concept system
utilizing Retrieval Augmented Generation (RAG) to enhance the reliability and contextual
accuracy of large language models (LLMs). It encompasses key processes such as data
preprocessing, embedding generation, and retrieval, all integrated into a structured pipeline
designed to handle unstructured text data efficiently.

The methodology leverages advanced libraries and tools like langchain, ChromaDB, and the
MXBAI embedding model, ensuring scalability and precision. It also incorporates techniques like
Recursive Text Splitting and cosine similarity ranking to optimize document processing and
context retrieval. Each phase of the methodology is crafted to align with the project’s objectives,
focusing on accurate information retrieval, relevance ranking, and delivering context-rich
responses. The subsequent sections provide a detailed breakdown of the technical and logical
underpinnings of this workflow.

### Software and Libraries
Programming Language: ![Python](https://github.com/arnab-raychaudhari/stock-aggregation-data-pipeline/blob/9f69abd4d570c16be5bc47b330bca84032e24aa5/logos--python.svg)<b><a href="https://www.python.org/about/" target="_blank">Python</a></b>

Python is a versatile, high level programming language known for its readability and vast
ecosystem of libraries. It is widely used in data science, web development, and machine learning
due to its simplicity and extensive community support (Van Rossum, 1995).

<b><a href="https://www.nltk.org/" target="_blank">Natural Language Toolkit (NLTK)</a></b>
Developed by researchers at the University of Pennsylvania, NLTK is a leading platform for
building Python programs to work with human language data. It provides tools for text
preprocessing, tokenization, stemming, and linguistic analysis, making it indispensable for natural
language processing (NLP) tasks (Bird et al., 2023).

<b><a href="https://github.com/chroma-core/chroma" target="_blank">ChromaDB</a></b>
ChromaDB is a modern vector database designed for storing and querying embeddings. It provides
a persistent storage solution for embeddings generated by machine learning models, enabling
efficient similarity searches and retrieval. Its integration with LLMs ensures seamless management
of document embeddings (Chroma, 2023).

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/f7e519c41411c0c9e29b964567819fdabf323662/langchain.svg" width="40" /><b><a href="https://www.gradio.app/" target="_blank"></a></b>
<b><a href="https://github.com/langchain-ai/langchain" target="_blank">LangChain</a></b>

LangChain is a framework specifically built for developing applications powered by large
language models (LLMs). It integrates components like document loaders, retrievers, and chains
for complex workflows. LangChain simplifies connecting LLMs to external knowledge bases and
retrieving contextually relevant information (LangChain, 2023).

<b><a href="https://www.sbert.net/" target="_blank">SentenceTransformers</a></b>
Developed by UKPLab at TU Darmstadt, Sentence Transformers is a Python library designed to
generate dense vector embeddings from textual data. These embeddings are crucial for tasks like
semantic search, clustering, and ranking. The library leverages pre-trained models such as BERT
and ROBERTA for high quality text representation (Reimers & Gurevych, 2023).

<img src="https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/29afb50abd87a325469a4bf8b7a6ca4b8fdfa643/gradio.svg" width="40" /><b><a href="https://www.gradio.app/" target="_blank">Gradio</a></b>

Gradio is an open-source Python library that simplifies the creation of user-friendly web interfaces
for machine learning models and data visualization. With drag and drop capabilities and
customizable themes, Gradio allows developers to test and demonstrate models interactively (Abid
et al., 2019).

<b>Ranking (Cross-Encoder Re-rankers)</b>
Ranking involves sorting documents based on their relevance to a given query. Cross-encoder re-
rankers, like HuggingFace’s BGE model, use transformer-based architectures to evaluate query
document relevance and reorder retrieved documents for improved contextual accuracy.

<b><img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original.svg" width="40"/><a href="https://matplotlib.org/" target="_blank">Matplotlib</a></b>
          
Matplotlib is a foundational library in Python for data visualization. It offers a wide range of chart
types, including line plots, bar graphs, and histograms, allowing developers to communicate data
insights effectively. Its flexibility makes it a staple in data science.

<b><a href="https://docs.python.org/3/library/concurrent.futures.html" target="_blank">Concurrent Processing (ThreadPoolExecutor)</a></b>
ThreadPoolExecutor, part of Python's concurrent. futures module, is used for parallelizing tasks to
enhance efficiency. It enables faster processing of large datasets by distributing workloads across
multiple threads, reducing bottlenecks in I/O-bound or computationally intensive operations.

<b><img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original.svg" width="40"/><a href="https://numpy.org/" target="_blank">Numpy</a></b>

NumPy is a powerful numerical computation library that provides support for arrays, matrices, and
mathematical functions. It is foundational for scientific computing in Python, offering optimized
operations for linear algebra, statistics, and data manipulation.

<b>Ollama</b>
Ollama is a platform designed to enable efficient deployment and use of large language models
(LLMs) such as Llama 3.1 on local machines. It simplifies the process of running sophisticated
LLMs by offering tools that optimize performance while reducing dependency on cloud-based
infrastructure (Paulo C. S. B., 2023).


<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" width="40"/><b>Virtual Machine</b>

The application was developed on an Amazon EC2 virtual machine, providing a robust and
scalable cloud environment for efficient processing. All the necessary libraries and packages,
including Python, Gradio, LangChain, ChromaDB, NLTK, and others, were pre-installed on the
virtual machine to establish a reliable foundation for development. This setup ensured that the
environment was optimized for deploying, testing, and refining the application while maintaining
compatibility with the various components of the RAG system. Due to the sensitive nature of the
information, we are not documenting the specific details of the virtual machine in this report,
ensuring the security and confidentiality of the development environment. The cloud based EC2
instance offered the flexibility and computational resources needed for seamless integration and
execution of complex workflows.

### Data Loading
In this project, document loading serves as the foundational step to prepare raw textual data for
further processing, including embedding generation and retrieval tasks. The process ensures
efficient reading, handling, and preprocessing of large datasets required for training and running
the RAG system. This step is critical for optimizing the performance of downstream tasks like
chunking, embedding creation, and query retrieval.

#### Importance of This Approach
1. Efficiency in Handling Large Datasets: Many RAG applications involve extensive textual data, which can take significant time to process if handled sequentially. By leveraging concurrent processing, the code drastically reduces the loading time.
2. Scalability: The use of ThreadPoolExecutor ensures the methodology can scale with the
increasing number of files or the complexity of datasets, a critical requirement for
enterprise grade applications.
3. Data Validation: Sampling ensures that the loaded data aligns with expectations before
proceeding to the more computationally expensive embedding and retrieval steps. This
minimizes errors at later stages.
4. Flexibility and Modularity: The modular functions (read_file and load_files_concurrently)
can be reused across different datasets, making the methodology adaptable to changes in
5. the project scope.
Robust Handling of Unstructured Data: The framework is designed to work seamlessly
with unstructured textual data, aligning with the project's goal of analyzing diverse
document types, such as public laws and farm bills.
6. Real-Time Feedback: Performance metrics like elapsed time and data size provide real-
time feedback, enabling developers to optimize system performance iteratively.
This process ensures that the document loading process is fast, reliable, and adaptable, aligning
with the high-performance requirements of a RAG application. By prioritizing scalability and
modularity, the approach allows seamless integration with other components of the system,
ensuring efficient processing of large and varied datasets. Figure 1 shows the code used in loading
the data .

<b>Figure 1: Data Loading Code Snippet</b>

![Figure 1: Data Loading Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/d0e8a56a049924163d6af10b504a3ede104f4ac2/Figure%201%20Data%20Loading%20Code%20Snippet.png)

### Chunking
In the text chunking process, the raw data is segmented into smaller, manageable pieces called
"chunks." This stage is essential for optimizing the processing of lengthy documents, enabling
efficient embeddings and retrieval in the RAG pipeline. Chunking is implemented using the
RecursiveCharacterTextSplitter from LangChain, combined with OpenAI's tiktoken encoder.
These tools work together to ensure each chunk is appropriately sized for downstream processing
while maintaining semantic coherence.

#### RecursiveCharacterTextSplitter
The RecursiveCharacterTextSplitter is a versatile tool designed to split long texts into smaller
chunks based on specified separators. For this project, the separators include ["\n\n", "\n", ". ", " ",
""], which prioritize splitting text at natural boundaries like paragraphs, sentences, and spaces.
This recursive approach ensures that text is split logically, retaining meaning and context within
each chunk. It also prevents abrupt truncation of sentences or important content.

Chunk Size: Set to 512 characters, this size aligns with typical tokenization limits for LLMs,
balancing granularity and context retention.

Chunk Overlap: Defined as 50 characters, overlap ensures that key information straddling chunk
boundaries is preserved. This is particularly important for maintaining semantic continuity and
improving retrieval quality.

#### <b><a href="https://github.com/openai/tiktoken" target="_blank">Tiktoken Encoder</a></b>
The tiktoken encoder, developed by OpenAI, is a tokenizer specifically designed for large
language models like GPT to enhance tokenization efficiency and compatibility with their LLM
architectures. It translates text into a sequence of tokens that the model can process. The integration
of tiktoken within the splitter ensures that the chunking respects the tokenization logic of the
underlying LLM, preventing errors related to token overflow or misalignment.

#### Why Chunking is Important?
1. Efficient Embeddings: Large documents can exceed token limits of models like GPT.
2. Chunking ensures that text is divided into processable segments.
Improved Retrieval: Smaller chunks allow more granular and precise information retrieval,
essential for the RAG framework.
3. Semantic Coherence: Recursive splitting with overlaps maintains the integrity of content,
ensuring each chunk is meaningful and retains context.
4. Performance Optimization: Processing smaller chunks reduces computational overhead
and prevents memory bottlenecks during embedding generation and model inference.

The combination of RecursiveCharacterTextSplitter and tiktoken is ideal for the project as it aligns
with the requirements of LLM tokenization and retrieval-based applications. By optimizing chunk
sizes and overlaps, this approach ensures the LLM receives high-quality input for embedding
generation and contextual retrieval. Figure 2 shows the code implemented to facilitate chunking.

<b>Figure 2: Chunking Code Snippet</b>

![Figure 2: Chunking Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/fc2209cd58d94f3ea65af778857f6b6e4d077701/Figure%202%20Chunking%20Code%20Snippet.png)

### Embedding Creation and ChromaDB
This step is crucial in preparing the system for accurate and contextually relevant document
retrieval, a core component of the RAG framework. Embedding generation transforms textual data
into numerical representations, enabling similarity-based searches in a high-dimensional vector
space. The embeddings are stored and managed using ChromaDB, a scalable vector database
optimized for efficient storage and retrieval of high-dimensional data.

#### Embedding Creation
Embeddings are numerical representations of text, allowing semantic information to be captured
in a vector format. For this project, embeddings were created using a custom embedding model
(embed_model). Here’s how the process works:
The embed_model (a SentenceTransformer-based model) processes the text of new or modified
documents. It generates embeddings, ensuring that semantically similar documents are close in the
vector space.

#### Why Embeddings are Necessary?
Embeddings enable similarity computations like cosine similarity or cosine distance to match user
queries with relevant documents. Unlike traditional keyword searches, embeddings capture the semantic essence of text, improving
retrieval accuracy.

#### Process Flow
Check for Existing Embeddings: The script first identifies which documents already have
embeddings stored in ChromaDB by comparing document IDs.

Embed Only New Documents: To optimize performance, only new or modified documents are
passed to the embedding model.

Batch Embedding: Embeddings are generated in batches, minimizing memory overhead while
ensuring parallel processing efficiency.

#### ChromaDB Integration
ChromaDB is a vector database designed to efficiently store, search, and retrieve embeddings. It
supports scalable management of high-dimensional vector data, making it ideal for RAG pipelines
where quick, similarity-based document retrieval is required.

##### Why ChromaDB is Ideal for This Project?
1. Persistence: Embeddings and associated metadata are stored persistently in a directory
(new_persist_directory), allowing the system to avoid re-computation across runs. This
saves computation time and makes the system efficient when scaling to larger datasets.
2. Efficient Retrieval: ChromaDB supports vector similarity operations, enabling quick
identification of documents that are semantically closest to the user’s query.
3. Flexibility: Collections in ChromaDB allow logical separation of embeddings based on
application domains (e.g., Farm_Bill, GAO, etc.).
4. Metadata and embeddings can be dynamically updated, ensuring relevance.

#### MXBAI embedding model
The MXBAI model (an embedding model developed by MixedBread AI) is leveraged for its
superior ability to generate high-dimensional embeddings optimized for tasks such as semantic
similarity and contextual information retrieval.
Here's why it is preferred:
1. High Semantic Fidelity: MXBAI embeddings are fine tuned for extracting nuanced
meanings from complex, unstructured text. This makes it highly suitable for domains like
federal contracting and legislative analysis, where context is critical.
2. Efficiency: MXBAI is designed to handle large scale datasets efficiently without
compromising the quality of the embeddings. This aligns well with the project's focus on
optimizing performance while dealing with vast collections of documents.
3. Specialized Use Cases: MXBAI has been tailored to perform well on tasks requiring high
precision and recall in document retrieval. For example, in comparing legislative bills, it
captures subtle policy differences that might otherwise go unnoticed.
Compatibility with ChromaDB: The model integrates seamlessly with ChromaDB’s vector
storage and retrieval system, ensuring persistent, scalable embeddings that can be reused
and updated as needed.

MXBAI excels in tasks such as clustering, ranking, and semantic search, which are central to
Retrieval Augmented Generation pipelines. These capabilities make it ideal for applications like
this project, where accurate information retrieval is paramount.

Overall Importance:
1. Operational Validation: By listing and sorting collections, the system ensures that it is
working with the most relevant and up to date embeddings.
2. Scalability: Persistent collections allow the system to scale and handle multiple use cases
without reprocessing embeddings unnecessarily.
3. Robustness: Handling errors and logging performance metrics ensure smooth operation
and debugging.
4. Optimized Retrieval: Using a precise embedding model like MXBAI maximizes the
effectiveness of document retrieval, enhancing the reliability of the RAG pipeline. Figure
3 shows code of how the MXBAI was implemented.

<b>Figure 3: MXBAI Embedding Code Snippet</b>

![Figure 3: MXBAI Embedding Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/fc2209cd58d94f3ea65af778857f6b6e4d077701/Figure%203%20MXBAI%20Embedding%20Code%20Snippet.png)

#### Cosine Distance and Cosine Similarity
Cosine similarity and cosine distance are metrics used to measure the orientation or angle
between two high dimensional vectors. These metrics are widely used in natural language
processing (NLP) and information retrieval tasks, particularly for comparing text embeddings
generated by models like MXBAI or Sentence Transformers. Figure 4 shows the code of how
the Cosine similarity was performed.

##### Cosine Similarity
Definition: Measures the cosine of the angle between two vectors. A value close to 1 indicates that
the vectors are pointing in the same direction (high similarity), while a value closer to 0 indicates
low similarity.
Interpretation: If two embeddings have a cosine similarity of 1, they are identical in terms of
direction (perfect semantic similarity). If the similarity is 0, they are orthogonal (completely
dissimilar).

##### Cosine Distance
A complementary metric derived from cosine similarity. It measures the dissimilarity between two
vectors, defined as:
Cosine Distance = 1−Cosine Similarity
Purpose: Highlights dissimilarity, making it useful for ranking or filtering out irrelevant data.

##### Logic Behind Using Cosine Similarity and Distance
1. Handling Embeddings: Document embeddings represent text in a high dimensional vector
space where semantically similar texts are closer to each other. Cosine similarity ensures
that the comparison focuses on orientation (semantic similarity), rather than magnitude,
which is irrelevant in NLP tasks.
2. Retrieval and Ranking: By calculating the cosine similarity between the query embedding
and document embeddings, we identify which documents are most relevant to the query.
Cosine distance provides an inverse measure to rank the least similar documents if needed.
3. Efficiency: By leveraging precomputed embeddings stored in ChromaDB, the system
avoids re-computation, improving runtime performance.
4. Scalability: This approach can handle thousands of embeddings efficiently, making it ideal
for large-scale applications.
5. Relevance Filtering: This method ensures only the most relevant document chunks are
retrieved and fed into the retrieval-augmented generation (RAG) system, enhancing
response accuracy.

<b>Figure 4: Cosine Similarity and Distance Code Snippet</b>

![Figure 4: Cosine Similarity and Distance Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/fc2209cd58d94f3ea65af778857f6b6e4d077701/Figure%204%20Cosine%20Similarity%20and%20Distance%20Code%20Snippet.png)

#### Initiating the Retriever and Reranker
The retriever is a critical component in the Retrieval Augmented Generation (RAG) framework,
responsible for extracting relevant document chunks from the vector database based on the user's
query. This code segment dynamically configures the retriever by determining the number of
document chunks (doc_chunks_to_retrieve) that should be retrieved from the database for further
processing.

##### Logic Behind Retriever Initialization
The number of document chunks to retrieve is calculated based on the conditions:
• If overwrite_doc_chunks is enabled, a pre-set value (doc_chunk_val) determines the
retrieval size, ensuring precise control over chunk retrieval.
• If the total number of chunks (len(chunks)) exceeds 200, a fixed number (100) is retrieved,
balancing computational efficiency with context depth.
• Otherwise, a proportional value (half the total chunks) is calculated for retrieval,
maintaining relevance without overloading the system.

##### Why Important?
Focused Contextual Retrieval: By selecting a precise number of document chunks, the retriever
ensures that the downstream language model is provided with only the most relevant context,
enhancing the accuracy of responses.
Efficiency: Dynamically determining doc_chunks_to_retrieve avoids overloading the system with
unnecessary data, optimizing retrieval time and memory usage.
Scalability: The logic is adaptable for large datasets, balancing the tradeoff between relevance and
computational load.
Re-Ranker Integration: When need_reranking is enabled, the retriever works in conjunction with
a ranking mechanism to prioritize the top documents, further refining the context for the RAG
pipeline.

#### ReRanking
The reranker implemented in the code uses the HuggingFaceCrossEncoder11, specifically the
model BAAI/bge-reranker-v2-m3. originates from the Beijing Academy of Artificial Intelligence
(BAAI), a leading research institution in AI. The institution is renowned for its contributions to
advancing deep learning techniques, particularly in natural language processing and model
optimization.

Reranking is a process in which an initial set of retrieved results is reorganized or reprioritized
based on their relevance to a given query. In the context of the code provided, reranking is
implemented to refine the selection of document chunks retrieved from a vector database. This
ensures that the most contextually relevant and high-quality results are prioritized for use by the
Retrieval Augmented Generation (RAG) framework. Figure 5 shows code implemented to
facilitate the re-ranking.

Cross-Encoder Reranker Initialization: A model (HuggingFaceCrossEncoder) is used to compute
relevance scores for each document in the retrieved set relative to the user query.
The CrossEncoderReranker assigns scores to each document query pair, enabling a more nuanced
understanding of relevance than traditional retrieval.

Compression Retriever: It integrates the reranker and base retriever, ensuring that only the most
relevant chunks are selected.

This combination reduces the computational load on the downstream language model by feeding
it a smaller, more targeted subset of documents.

##### Why Important?
1. Improved Relevance: Reranking evaluates the retrieved documents against the query using
a more sophisticated model (cross-encoder) that understands both context and
relationships. This ensures that the results are not just approximate matches but highly
relevant to the user's needs.
2. Enhances LLM Performance: Large Language Models (LLMs) like GPT rely heavily on
the context provided during inference. Feeding the model with highly relevant and curated
data improves the quality, accuracy, and reliability of the generated responses.
3. Contextual Prioritization: In scenarios where multiple documents are retrieved, not all are
equally important. Reranking helps prioritize the documents that provide the most value,
ensuring that critical information is not overlooked.
4. Scalability and Efficiency: By reducing the number of documents passed to the LLM,
reranking lowers computational overhead. This is particularly important in large-scale
applications where both the number of documents and the complexity of queries can be
significant.
5. Mitigating Noise: Raw retrieval often includes documents that are less relevant or contain
peripheral information. Reranking filters out this noise, focusing on documents that directly
address the user's query.
6. User Experience: Reranked results lead to more concise, precise, and contextually accurate
answers, enhancing the overall user experience.

<b>Figure 5: Re-ranking Code Snippet</b>

![Figure 5: Re-ranking Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/fc2209cd58d94f3ea65af778857f6b6e4d077701/Figure%205%20Re-ranking%20Code%20Snippet.png)

#### Invoking the Retrieval Chain

Invoking the retrieval chain to generate responses based on the input query (user_prompt) and the
retrieved context. It includes the flexibility to either augment responses with context retrieved
using Retrieval Augmented Generation (RAG) or bypass the RAG step when reranking is not
required or when contextual documents are absent. Here’s why this flexibility is important:

Conditional RAG-Augmented Response:
If need_reranking == "Yes", the process ensures that the most contextually relevant documents are
identified and reranked before feeding them to the retrieval chain for response generation.
If need_reranking == "No", the retrieval chain uses the documents retrieved directly from the
vector database without additional reranking.

##### Why Flexibility is Important?
1. Addressing Context Limited Scenarios: When no relevant documents are retrieved from
the vector database, RAG might fail to produce a meaningful response. In such cases,
bypassing RAG allows the model to answer the query based on its internal pre-trained
knowledge, ensuring a response is generated even without external context.
2. Streamlining for Simpler Queries: Not all queries require extensive contextual grounding.
For instance, straightforward or generalized queries may not benefit from RAG, and
bypassing it can save computational resources and time.
3. Balancing Accuracy and Performance: RAG adds a layer of accuracy by grounding
4. responses in external data but can increase latency. By enabling a "no RAG" path, the
system remains responsive for queries that prioritize speed over exhaustive context.
Fallback Mechanism for Missing Context: In cases where the retrieval system fails to locate
relevant documents, the flexibility ensures that the model doesn’t rely solely on absent or
irrelevant data. This fallback mechanism avoids generating hallucinated responses based
on incorrect context.
5. Resource Efficiency: Bypassing unnecessary processes (like RAG) for certain queries
reduces computational overhead and speeds up response generation.
6. Versatility in Applications: Systems employing this methodology can adapt to varied use
cases, from knowledge-intensive tasks like legislative analysis to general purpose
querying, without compromising flexibility.

By including this flexibility, the system ensures both robustness and adaptability, maintaining high
performance while avoiding the pitfalls of over-reliance on RAG when context is absent or
irrelevant. Figure 6 shows the code implemented to initiate the retrieval chain.

<b>Figure 6: Invoking the Retrieval Chain Code Snippet</b>

![Figure 6: Invoking the Retrieval Chain Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/fc2209cd58d94f3ea65af778857f6b6e4d077701/Figure%206%20Invoking%20the%20Retrieval%20Chain%20Code%20Snippet.png)

#### The Online Mode
The Online Mode section defines a dictionary named prompts that categorizes pre-defined queries
or questions for specific application domains. These prompts are designed to address domain
specific queries and form a critical part of the interaction between the user and the system.

##### Logic behind the Online Mode
1. Domain Specific Queries: Online Mode organizes pre-written queries for different
application domains like "Farm_Bill", "USSP_Light", "GAO", and "USSP". This allows
users to quickly access relevant questions for their selected domain without manually
creating queries.
2. Streamlined Query Selection: By offering categorized prompts, the system enhances
usability and efficiency, particularly for users unfamiliar with technical or domain specific
language.
3. Focus on Real World Application:
The prompts cover a broad range of topics, including:
I. Legislative comparisons (e.g., Farm Bills).
II. Federal spending trends and appropriations (e.g., USSP).
III. Evaluation of federal program effectiveness (e.g., GAO).

These questions ensure that the system is aligned with real world analytical needs. Figure 7 shows
online mode code implemented.

<b>Figure 7: Online Mode Code Snippet</b>
![Figure 7: Online Mode Code Snippet](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/08a59fcc77eb680c84bf4a4b4ce08db3b892a2bc/Figure%207%20Online%20Mode%20Code%20Snippet.png)

#### Gradio Interface
Gradio is an open-source Python library that enables the rapid development and deployment of
machine learning (ML) models and applications through interactive web-based interfaces. It
simplifies the user experience by creating highly customizable graphical interfaces for ML
applications without the need for specialized front end skills. Its ease of integration, accessibility
via a web browser, and flexibility make it an essential tool for showcasing and deploying ML
powered solutions.

#### Core Features of the Insight Engine

##### Dynamic Dropdowns for Application and LLM Selection
Why? Different applications (Farm_Bill, USSP_Light, and GAO) address unique datasets and
analytical needs. The dropdown ensures that the interface dynamically adapts to these use cases.
How They Help:

• Simplifies navigation for users by isolating relevant tools and prompts.
• Encourages modularity, reducing clutter and confusion.
• Allows for targeted responses, enhancing the system’s focus and efficiency.

