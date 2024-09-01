**Custom Chatbot Using Langchain**
Project Overview
This project demonstrates the creation of a custom chatbot using Langchain. The chatbot is designed to interact with users by providing information on technical courses extracted from the website Brainlox. 

The solution involves the following steps:

Data Extraction: Utilizing URL loaders from Langchain to scrape data from the specified website, including course titles, descriptions, prices, and links.

Embedding Creation: Generating embeddings for course descriptions using a pre-trained model. These embeddings are stored in a vector store to facilitate efficient and accurate querying.

Flask RESTful API: Developing a Flask-based RESTful API to handle conversations with the chatbot. The API processes user queries, retrieves relevant information from the vector store, and generates responses.

Installation and Setup

Install Dependencies: Make sure to set up a virtual environment and install the necessary packages.

`pip install -r requirements.txt`

Run the Flask API:

`python app.py`

Access the API: The API will be available at http://127.0.0.1:5000.

Files Included

app.py: Flask application code for the RESTful API.
index.faiss: FAISS index file for the vector store.
index.pkl: Pickle file for additional vector store metadata.
development.py: Colab script used for data extraction, embedding creation, and vector store setup.
requirements.txt: List of required Python packages to run the project.
