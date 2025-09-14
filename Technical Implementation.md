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

