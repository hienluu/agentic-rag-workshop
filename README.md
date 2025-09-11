# Agentic RAG Workshop
Instructions and resource for the Agentic RAG workshop

### Prerequites
* Generate a Hugging Face [Access Token](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-token) and use it to login from Colab
  * Store it in [Google Colab secrets](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb#scrollTo=mhFKmRmxi5B-) with name HF_TOKEN
* GROQ_API_KEY
  * Go to [Groq console](https://console.groq.com/playground) to create an API key
* GEMINI_API_KEY
  * Go to [Gemini AI Studio](https://aistudio.google.com/apikey) to create an API key

### Part 1 - Naive RAG
* [Chunk Visualizer - with length function](https://huggingface.co/spaces/m-ric/chunk_visualizer)
* [0-qconsf-chunking](https://colab.research.google.com/drive/16S9YG3CgTwu8XzkNAhwn9b4mIjmR3Cl-#scrollTo=Z5ozDZKMRveJ)
* [qconsf-embedding](https://colab.research.google.com/drive/1WY_3S6-vQyCRTQkud4y0wHlhqr7oEiId#scrollTo=lVTCg-A7Pasc)
* [qconsf-basic-rag-with-gradio](https://colab.research.google.com/drive/1F0JZhjYaA8ynq7EirxiMVZWEzxeTX9cY#scrollTo=LVfwhDNuq-_E)


### Resources:
* [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks - 2020](https://arxiv.org/abs/2005.11401)
* [From Local to Global: A Graph RAG Approach to Query-Focused Summarization - 2025](https://arxiv.org/abs/2404.16130)
* [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779)
  * [github repo](https://lightrag.github.io/), [video](https://www.youtube.com/watch?v=oageL-1I0GE), [local LightRAG video](https://www.youtube.com/watch?v=g21royNJ4fw)
* [Generative Benchmarking from ChromaDB](https://research.trychroma.com/generative-benchmarking)
  * [Github repo](https://github.com/chroma-core/generative-benchmarking)
* [Systematically Improving RAG Applications](https://github.com/567-labs/systematically-improving-rag)
* [Full Stack Retrieval](https://community.fullstackretrieval.com/)
