# Class:        CS 7375/W02

# Term:         Spring 2025

# Instructors:  Dr. Rasael Mahmud, Dr. Coskun Cetinkaya

# Project:      Personalizing and Optimizing Learning Experiences using AI

import requests
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# This program helps students find relevant learning resources on Microsoft Learn.
# It grabs course data then processes it with NLP tools, and recommends content based on user input.
# After processing it creates a course program compose of five courses, a main course that is the closes
# one related to the student's query and four more recommended courses that are similar to the core course.
# ------------------------------------------

# Ensure the sentence tokenizer from NLTK is ready to use and downloads it if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("NLTK punkt tokenizer not found. Downloading...")
    nltk.download("punkt")
    print("NLTK punkt tokenizer downloaded successfully.")

# TEXT PROCESSING HELPERS

def summarize_text(text, num_sentences=6):
    # Shortens long text into a brief summary using the first few meaningful sentences
    # Converts sentences into a cohesive paragraph
    if not text or not isinstance(text, str):
        # If the input isn't valid text it returns a default message
        return "No description available."
    try:
        # Initialize the sentence tokenizer
        tokenizer = PunktSentenceTokenizer()
        # Get rid of unnecessary whitespace and tidy up the text
        clean_text = text.strip().replace("\n", " ").replace("\r", " ")
        # Tokenize the text into sentences
        sentences = tokenizer.tokenize(clean_text)
        # Skip very short sentences that don't add to the context
        sentences = [s for s in sentences if len(s.split()) > 3] # Keep sentence with more than 3 words
        # Return the summary and if something goes wrong during summarization, show a shortened version of the original
        return " ".join(sentences[:num_sentences]) if sentences else "No description available."
    except Exception as e:
        # Print error message in case summarization fails
        print(f" Error during text summarization: {e}")
        # Returns the original text if the summarization were to fail
        return text[:500] + "..." if len(text) > 500 else text

# SEMANTIC FILTERING

def semantic_filter(query, items, threshold=0.05):
    # Uses semantic similarity to keep content that's closely related to what the user's query
    if not items:
        return []

    # Put together title and description so we have more context for each item
    texts = [f"{i.get('title', '')} {i.get('description', '')}" for i in items]
    # The query goes first in the list so we can compare everything to it
    corpus = [query] + texts
    try:
        # Tries to calculate similarity scores using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer()
        # Calculates cosine similarity
        vectors = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        # Keeps items that are similar enough, and sort them by how closely they match
        # Filter items based on threshold
        results = [(item, score) for item, score in zip(items, scores) if score > threshold]
        # Sorted filtered items by their score
        sorted_items = sorted(results, key=lambda x: x[1], reverse=True)
        # Display how many items passed the filter
        print(f"Found {len(sorted_items)} relevant items (threshold: {threshold}).")
        # Print the top 10 items with their scores
        print("\n--- Top Relevant Items with Relevancy Score, 0 for no relevancy to 1 being identical shown in decimal ---")
        for item, score in sorted_items[:10]:
            # Format the score to a few decimal places and print program segment with core course title
            print(f"  Score: {score:.4f} - Program Segment: {item.get('title', 'N/A')}")
        print("-" * 50 + "\n")
        # Items without scares are sorted by relevance
        return [item for item, _ in sorted_items]
    
    except Exception as e:
        # Fall back to a simple keyword match if the calculation fails
        print(f" TF-IDF similarity calculation failed: {e}. Using simple keyword matching.")
        q_lower = query.lower()
        return [
            i for i in items
            if q_lower in i.get('title', '').lower() or q_lower in i.get('description', '').lower()
        ]

def recommend_similar_courses(base_course, course_list, top_n=5):
    # Recommends courses that are similar in content to a core course
    if not base_course or not course_list:
        return []

    # Build a sentence description for each course so we can compare them
    reference_text = f"{base_course.get('title', '')} {base_course.get('description', '')}"
    # Skip comparing the course to itself
    others = [c for c in course_list if c.get('url') != base_course.get('url')]
    if not others:
        return []
    # Prepares text for comparison starting with the base course description, 
    # followed by the combined titles and descriptions from recommended courses
    corpus = [reference_text] + [f"{c.get('title', '')} {c.get('description', '')}" for c in others]
    try:
        # TF-IDF converts text into numerical vectors assigned by word importance
        vectorizer = TfidfVectorizer()
        # Vectorizer analyzes vocabulary and word frequncy and creates a vector for ea.
        vectors = vectorizer.fit_transform(corpus)
        # Calculate similarity scores, vectors[0:1] being the base course 
        # and vectors[1:] the recommended courses  
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        # Only keep the top matches and excludes those with a low similarity
        ranked = sorted(
            [(c, s) for c, s in zip(others, scores) if s > 0.01],
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return [course for course, _ in ranked]
    except Exception as e:
        # If anything fails during comparison, return an empty list to avoid crashing
        print(f" Failed to generate course recommendations: {e}")
        return []

# API LOOKUP

def search_microsoft_learn_resources(query):
    # Looks up Microsoft Learn API and returns content related to the user's search query
    base_url = "https://learn.microsoft.com/api/catalog"
    results = []

    print("Searching Microsoft Learn...")
    try:
        # Pulls data from the Microsoft Learn API
        response = requests.get(base_url, timeout=20)
        # Check if the request was successful
        response.raise_for_status()
        # Parse JSON Data
        data = response.json()
        # Clean up and normalize the entries from both modules and learning paths
        entries = data.get("modules", []) + data.get("learningPaths", [])
        print(f"Found {len(entries)} total items from MS Learn catalog.")

        # Process each item fetched from the API
        for entry in entries:
            # Ignore entries without a title or link
            if not isinstance(entry, dict):
                continue

            # Extracts relevant fields
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            url = entry.get("url", "")
            locale = entry.get("locale", "en-us")

            # Skips missing info
            if not title or not url:
                continue

            # Fix relative links by making them full URLs
            if url.startswith("/"):
                absolute_url = f"https://learn.microsoft.com{url}"
            elif url.startswith("http"):
                absolute_url = url
            else:
                absolute_url = f"https://learn.microsoft.com/{locale}/{url.lstrip('/')}"

            # Fills out relevant fields from extraction
            results.append({
                "title": title,
                "description": summary or "No description provided.",
                "url": absolute_url,
                "source": "Microsoft Learn"
            })

    except requests.exceptions.RequestException as e:
        # If there's a problem communicating to the API, print an error and stop
        print(f" Error fetching data from Microsoft Learn API: {e}")
        return []
    except (ValueError, KeyError, TypeError) as e:
        # Don't proceed if the data format is broken 
        print(f" Error processing the catalog data: {e}")
        return []
    except Exception as e:
        print(f" An unexpected error occurred: {e}")
        return []

    if not results:
        print(" No processable entries found in the catalog data.")
        return []

    # If everything looks good, filter the results based on how relevant they are to the query
    print(f"Ranking results for relevance to '{query}'...")
    return semantic_filter(query, results)

# MAIN EXECUTION
if __name__ == "__main__":
    # Ask the user what topic they're interested in
    query = input("Enter a topic to search in Microsoft Learn: ")
    # Perform the search and filtering
    results = search_microsoft_learn_resources(query)

    # Announce if no relevant data was found
    if not results:
        print("\nNo relevant resources were found for your query.")
    else:
        # Paginate so it's easier to read and navigate
        index = 0
        page_size = 10
        # Loop that controls pagination
        while index < len(results):
            # Get and print initial page of results
            current_batch = results[index:index + page_size]
            print(f"\n--- Displaying Microsoft Learn Program Segments {index + 1} - {index + len(current_batch)} of {len(results)} ---")

            # Process and print ea. course in the page
            for i, resource in enumerate(current_batch, index + 1):
                print(f"\n--- Program Segment {i} ---")
                # Add up the descriptions of the main and related content to get a better summary
                print("  Analysing related courses...")
                recommendations = recommend_similar_courses(resource, results, top_n=4)
                # Combine the description of the initial resource and its recommendations
                combined_text = resource.get('description', '')
                # Add descriptions of recommended courses
                for r in recommendations:
                    combined_text += " " + r.get('description', '')
                # Put together a combined summary of all courses into a paragraph
                summary = summarize_text(combined_text.strip())

                # Print a summary that combines the main resource with its suggestions
                print("\n*** Course Program Overview ***")
                print(f"   {summary}")
                print("-" * 20)

                # Print details of the main resource and any similar ones
                print("  Included Materials:")
                print(f"      Core Course: {resource.get('title')}")
                print(f"      Description: {resource.get('description')}")
                print(f"      URL: {resource.get('url')}\n")
                if recommendations:
                    print("    Related Courses:")
                    for idx, r in enumerate(recommendations, 1):
                        print(f"      {idx}. {r.get('title')}")
                        print(f"         Description: {r.get('description')}")
                        print(f"         URL: {r.get('url')}")
                else:
                    print("    No related courses were found.")

                print("=" * 50)

            index += page_size
            # Ask the user if they want to continue browsing results
            if index < len(results):
                cont = input("\nShow more results? (y/n): ").strip().lower()
                # Stop the program if the user doesn't enter 'y'
                if cont != 'y':
                    print("Exiting.")
                    break
            else:
                print("\n--- End of results ---")
