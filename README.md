# Microsoft Learn Semantic Search & Recommender

## Project Context

This script was developed as part of the **"AI Personalizing and Optimizing Learning Experiences"** project at **Kennesaw State University**.

**Group Members:** Mauricio Gonzalez, Cole Peterson, Faizan Shaikh

## Description

This Python script (\`Microsoft\_AIsearch.py\`) interacts with the Microsoft Learn Catalog API to find learning resources (modules and learning paths) related to a user-provided topic. It then processes these results using Natural Language Processing (NLP) techniques:

1.  **Fetches Data:** Retrieves the entire catalog from the Microsoft Learn API.  
2.  **Semantic Filtering:** Uses TF-IDF and Cosine Similarity (\`scikit-learn\`) to filter the fetched resources based on semantic relevance to the user's query.  
3.  **Recommendation:** For each top-filtered result, it finds other semantically similar courses within the filtered set to suggest related learning materials.  
4.  **Summarization:** Uses NLTK to generate a paragraph summarizing the content of the core course and its recommendations, combining their descriptions.  
5.  **Display:** Presents the results in "Program Segments," showing the generated overview, a list of included resources, and includes pagination to handle potentially large result sets.

## Requirements

*   Python 3.x  
*   Libraries listed in \`requirements.txt\` (\`requests\`, \`nltk\`, \`scikit-learn\`, etc.)

## Installation

1.  **Clone the repository or download the script files** (\`Microsoft\_AIsearch.py\`, \`requirements.txt\`, etc.) into a directory.

2.  **Create and activate a virtual environment (Recommended):**  
    ```bash  
    \# Windows  
    python \-m venv venv  
    .\\venv\\Scripts\\activate

    \# macOS/Linux  
    python3 \-m venv venv  
    source venv/bin/activate  
    \`\`\`

3.  **Install required Python libraries:**  
    Navigate to the directory containing the files in your terminal and run:  
    ```bash  
    pip install \-r requirements.txt  
    \`\`\`

4.  **Download NLTK data:**  
    The script requires the 'punkt' sentence tokenizer data from NLTK. It includes code to automatically attempt this download on the first run if the data is not found. If you encounter issues or prefer to download it manually beforehand, open a Python interpreter (\`python\` or \`python3\` in your terminal) within the activated virtual environment and run:  
    ```python  
    import nltk  
    nltk.download('punkt')  
    exit()  
    ```

##Copyright and Licensing  
Copyright (c) 2025 Mauricio Gonzalez

Permission is hereby granted, free of charge, to the following individuals only:

Mauricio Gonzalez (Author)  
Cole Peterson  
Faizan Shaikh  
to use, copy, and modify this software and associated documentation files (the "Software") solely for the purpose of the Kennesaw State University project titled "AI Personalizing and Optimizing Learning Experiences".

Use, modification, distribution, or sublicensing of the Software outside the scope of this specific university project or by individuals not listed above is strictly prohibited without the express written permission of the copyright holder Mauricio Gonzalez.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Usage

Run the main script from your terminal within the activated virtual environment:

```bash  
python Microsoft\_AIsearch.py

