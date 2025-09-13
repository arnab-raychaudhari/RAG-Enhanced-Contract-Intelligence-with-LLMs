# RAGov-contract-analysis
Leveraging AI to Enhance Data Driven Insights for Government Contracts

## Capstone Engagement
This project was undertaken, executed, and completed as part of the Master of Science in Business Analytics program at George Washington University. As a requirement of the degree, I was tasked with partnering with a real-world business (my client) to address an ongoing problem using advanced Artificial Intelligence techniques. The project spanned four months and was successfully delivered to the client in December 2024.

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
Programming Language: ![Python](https://github.com/arnab-raychaudhari/stock-aggregation-data-pipeline/blob/9f69abd4d570c16be5bc47b330bca84032e24aa5/logos--python.svg)Python

Python is a versatile, high level programming language known for its readability and vast
ecosystem of libraries. It is widely used in data science, web development, and machine learning
due to its simplicity and extensive community support (Van Rossum, 1995).

[NLTK](https://github.com/arnab-raychaudhari/RAGov-contract-analysis/blob/815a1c04a7d4181f61ba9be0905f2aef357d78f2/logo--nltk.webp)NLTK
Developed by researchers at the University of Pennsylvania, NLTK is a leading platform for
building Python programs to work with human language data. It provides tools for text
preprocessing, tokenization, stemming, and linguistic analysis, making it indispensable for natural
language processing (NLP) tasks (Bird et al., 2023).



