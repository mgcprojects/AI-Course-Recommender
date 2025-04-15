import requests
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- NLTK: Download punkt tokenizer if not available ---
# The 'punkt' tokenizer is needed by NLTK to split text into sentences.
# This block checks if it's already downloaded and downloads it if necessary.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("⬇️ Downloading NLTK 'punkt' tokenizer...")
    nltk.download("punkt")
    print("✅ NLTK 'punkt' downloaded.")

# --- AI/NLP Helpers ---

def summarize_text(text, num_sentences=6):
    """
    Summarizes the input text to approximately paragraph length (default 6 sentences)
    using NLTK's Punkt sentence tokenizer.

    Args:
        text (str): The text to summarize.
        num_sentences (int): The target number of sentences for the summary.

    Returns:
        str: The summarized text, or a fallback message/text if summarization fails or input is invalid.
    """
    # Basic input validation
    if not text or not isinstance(text, str):
        return "No description available."
    try:
        # Initialize the sentence tokenizer
        tokenizer = PunktSentenceTokenizer()
        # Basic text cleaning: remove leading/trailing whitespace and replace newlines/carriage returns with spaces
        clean_text = text.strip().replace("\n", " ").replace("\r", " ")
        # Tokenize the cleaned text into sentences
        sentences = tokenizer.tokenize(clean_text)
        # Filter out very short sentences that are likely noise or formatting remnants
        sentences = [s for s in sentences if len(s.split()) > 3] # Keep sentences with more than 3 words
        # Join the first 'num_sentences' sentences, or all sentences if the text is shorter
        summary = " ".join(sentences[:num_sentences]) if len(sentences) > num_sentences else " ".join(sentences)
        # Return the summary, ensuring it's not empty or otherwise return a default message
        return summary if summary else "No description available."
    except Exception as e:
        # Handle potential errors during tokenization or processing
        print(f" Error summarizing text: {e}")
        # Fallback: return the beginning of the original text if summarization fails
        return text[:500] + "..." if len(text) > 500 else text

def semantic_filter(query, items, threshold=0.05):
    """
    Filters a list of items based on their semantic similarity to a given query,
    using TF-IDF and cosine similarity.

    Args:
        query (str): The search query text.
        items (list): A list of dictionaries, where each dictionary represents an item
                      and should have 'title' and 'description' keys.
        threshold (float): The minimum cosine similarity score for an item to be included.

    Returns:
        list: A list of items sorted by similarity score (descending) that meet the threshold.
              Returns fallback keyword matches if TF-IDF fails.
    """
    # Handle empty input list
    if not items:
        return []
    # Prepare documents for TF-IDF and combine title and description for each item
    documents = [f"{item.get('title', '')} {item.get('description', '')}" for item in items]
    # Create the corpus including the query at the beginning
    corpus = [query] + documents
    try:
        # Initialize and fit the TF-IDF vectorizer
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        # Calculate cosine similarity between the query vector (first row) and all item vectors
        similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

        # Pair items with their calculated similarity scores
        scored_items = list(zip(items, similarities))
        # Filter items based on the similarity threshold
        filtered_items = [item for item, score in scored_items if score > threshold]

        # Sort the filtered items by their similarity score in descending order
        # This re-finds the score for sorting, which is slightly inefficient but works.
        # A more efficient way would be to sort scored_items before extracting just the items.
        top_matches = sorted(
            filtered_items,
            key=lambda item: next(score for i, score in scored_items if i == item),
            reverse=True
        )

        print(f"🧠 Found {len(top_matches)} items above similarity threshold {threshold}.")
        return top_matches

    except Exception as e:
        # Handle potential errors during vectorization or similarity calculation
        print(f" Error during semantic filtering: {e}")
        # Fallback mechanism to perform simple case-insensitive keyword matching
        print(" Falling back to simple keyword matching.")
        query_lower = query.lower()
        fallback_matches = [
            item for item in items
            if query_lower in item.get('title', '').lower() or query_lower in item.get('description', '').lower()
        ]
        return fallback_matches

def recommend_similar_courses(selected_course, all_courses, top_n=5):
    """
    Recommends courses from a list that are semantically similar to a selected course.

    Args:
        selected_course (dict): The course to find recommendations for.
        all_courses (list): The pool of courses to recommend from.
        top_n (int): The maximum number of recommendations to return.

    Returns:
        list: A list of recommended course dictionaries, sorted by similarity.
    """
    # Handle invalid inputs
    if not selected_course or not all_courses:
        return []

    # Create the base text from the selected course's title and description
    base_text = f"{selected_course.get('title', '')} {selected_course.get('description', '')}"
    # Create a list of candidate courses, excluding the selected course itself identified by URL
    candidates = [
        c for c in all_courses
        if isinstance(c, dict) and c.get('url') != selected_course.get('url')
    ]

    # If no candidates are left return an empty list
    if not candidates:
        return []

    # Create the corpus with the base text followed by the text of candidate courses
    corpus = [base_text] + [f"{c.get('title', '')} {c.get('description', '')}" for c in candidates]

    try:
        # Vectorize the corpus using TF-IDF
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        # Calculate cosine similarity between the base text vector and all candidate vectors
        similarity_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
        # Pair candidates with their similarity scores
        scored_candidates = list(zip(candidates, similarity_scores))

        # Filter out candidates with very low similarity < 0.01 and sort by score
        top_recommendations = sorted(
            [cand for cand in scored_candidates if cand[1] > 0.01], # Filter low scores
            key=lambda x: x[1], # Sort by score
            reverse=True # Highest score first
        )[:top_n] # Take only the top N recommendations

        # Return only the course dictionaries from the sorted list
        return [rec[0] for rec in top_recommendations]

    except Exception as e:
        # Handle potential errors during recommendation process
        print(f" Error recommending similar courses: {e}")
        return []

def search_microsoft_learn_resources(query):
    """
    Fetches and processes learning resources, modules and learning paths
    from the Microsoft Learn Catalog API based on a query.

    Args:
        query (str): The search term.

    Returns:
        A list of resource found and filtered by semantic similarity.
    """
    # API endpoint
    base_url = "https://learn.microsoft.com/api/catalog"
    all_resources = []
    fetched_items_count = 0

    print("🔍 Searching Microsoft Learn...")
    try:
        # Make the GET request to the API
        response = requests.get(base_url, timeout=20) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Parse the JSON response
        catalog = response.json()
        # Combine items from "modules" and "learningPaths" sections of the catalog
        items = catalog.get("modules", []) + catalog.get("learningPaths", [])
        fetched_items_count = len(items)
        print(f"📚 Fetched {fetched_items_count} total items from MS Learn catalog.")

        # Process each item fetched from the API
        for item in items:
            # Basic validation to ensure item is a dictionary
            if not isinstance(item, dict):
                continue

            # Extract relevant fields, providing defaults if keys are missing
            title = item.get("title", "")
            description = item.get("summary", "") # Description field
            url = item.get("url", "")
            locale = item.get("locale", "en-us") # Default locale

            # Skip items missing essential information (title or URL)
            if not title or not url:
                continue

            # Construct the absolute URL if the provided URL is relative
            if url.startswith('/'):
                full_url = f"https://learn.microsoft.com{url}"
            elif url.startswith("http"):
                 full_url = url # Already absolute
            else:
                 # Attempt to construct URL assuming it's relative to locale
                 full_url = f"https://learn.microsoft.com/{locale}/{url.lstrip('/')}"

            # Collect the processed and standardized resource to the list
            all_resources.append({
                "title": title,
                "description": description if description else "No description provided.", # Ensure description is not None
                "url": full_url,
                "source": "Microsoft Learn" # Add source identifier
            })

    # Handle potential errors during the API request
    except requests.exceptions.RequestException as e:
        print(f" Error fetching Microsoft Learn content: {e}")
        return []
    # Handle potential errors during JSON parsing or data extraction
    except (ValueError, KeyError, TypeError) as e:
        print(f" Error parsing Microsoft Learn catalog: {e}")
        return []
    # Handle any other unexpected errors
    except Exception as e:
        print(f" An unexpected error occurred during fetching: {e}")
        return []

    # If no resources could be processed, return empty list
    if not all_resources:
        print("No resources could be processed from Microsoft Learn.")
        return []

    # Apply semantic filtering to the fetched resources based on the user's query
    print(f" Running semantic search for '{query}' on {len(all_resources)} processed resources...")
    filtered_results = semantic_filter(query, all_resources)

    print(f" Found {len(filtered_results)} potentially relevant items after semantic filtering.")
    return filtered_results

# --- Main Execution Block ---
# This code runs only when the script is executed directly and not imported as a module
if __name__ == "__main__":
    # Get the search topic from the user
    query = input("🔍 Enter a topic to search in Microsoft Learn: ")
    # Perform the search and filtering
    results = search_microsoft_learn_resources(query)

    # Handle case where no results are found
    if not results:
        print("\n No matching courses found for your query after filtering.")
    else:
        # --- Pagination Logic ---
        index = 0 # Current starting index for displaying results
        page_size = 10 # Number of results to display per page
        more_results = True # Flag to control the pagination loop

        # Loop to display results in pages
        while more_results and index < len(results):
            # Get the current batch (page) of results
            batch = results[index:index + page_size]
            print(f"\n--- Displaying Microsoft Learn Program Segments {index + 1} to {index + len(batch)} ---")

            # Use the full list of filtered results as the pool for recommendations
            recommendation_pool = results

            # Process and display each resource in the current batch
            for i, initial_resource in enumerate(batch, index + 1):
                print(f"\n--- Program Segment {i} ---")
                # Find courses similar to the current 'initial_resource'
                print("  Analysing related courses...")
                recs = recommend_similar_courses(initial_resource, recommendation_pool, top_n=4) # Get up to 4 recommendations

                # --- Generate Program Overview ---
                # Combine the description of the initial resource and its recommendations
                combined_text = initial_resource.get('description', '')
                if recs:
                    # Add descriptions of recommended courses, ensuring they exist
                    valid_rec_descriptions = [rec.get('description', '') for rec in recs if rec.get('description')]
                    if valid_rec_descriptions:
                         combined_text += " " + " ".join(valid_rec_descriptions)


                # Summarize the combined text into a paragraph
                program_summary = summarize_text(combined_text.strip(), num_sentences=6)

                # Print the generated overview
                print(f"\n*** Course Program Overview ***")
                print(f"   {program_summary}") # Indented summary
                print("-" * 20) # Separator

                # --- List Included Resources ---
                print("  Included Resources:")
                # Print details of the core initial course
                print(f"    - Core Course: {initial_resource.get('title', 'N/A')}\n      Description: {initial_resource.get('description', 'No description provided.')}\n      URL: {initial_resource.get('url', 'N/A')}")
                # Print details of the related recommended courses
                if recs:
                    print(f"    - Related Courses ({len(recs)}):")
                    for rec_idx, rec in enumerate(recs, 1):
                        print(f"      {rec_idx}. {rec.get('title', 'N/A')}\n         Description: {rec.get('description', 'No description provided.')}\n         URL: {rec.get('url', 'N/A')}")
                else:
                    # Message if no recommendations were found
                    print("    - No additional related courses found for this segment.")

                print("=" * 50) # Separator between program segments

            # Move to the next page index
            index += page_size
            # Check if there are more results to display
            if index < len(results):
                # Ask the user if they want to see the next page
                user_input = input("\nWould you like to see the next 10 results? (Y/N): ").strip().lower()
                # Stop if the user doesn't enter 'y'
                if user_input != 'y':
                    more_results = False
                    print("Exiting program.")
            else:
                # All results have been shown
                print("All results have been displayed.")
                # No need to set more_results = False, the loop condition (index < len(results)) handles this
