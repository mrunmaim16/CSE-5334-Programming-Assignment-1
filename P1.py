import time
import os
import math
# Uncomment the following line to download the necessary NLTK resources
# import nltk
# nltk.download()
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

start_time = time.time()
# Path to the corpus of presidential addresses
corpusroot = './US_Inaugural_Addresses'  
# Tokenizer, stopword removal, and stemming
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()

# Structure to hold TF-IDF values
tf_idf_vectors = {}
doc_count = defaultdict(int)  # Document frequency for IDF calculation

def preprocess_text(doc):
    # Tokenizes, removes stopwords, and stems the input document. 
    tokens = tokenizer.tokenize(doc.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return filtered_tokens

def compute_tf_idf():
    # Computes the TF-IDF vectors for the corpus. 
    global tf_idf_vectors
    file_list = os.listdir(corpusroot)
    total_docs = len(file_list)

    for filename in file_list:
        if filename.endswith('.txt'):
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                doc = file.read()
                tokens = preprocess_text(doc)

                # Calculate term frequency (TF)
                tf = defaultdict(int)
                for token in tokens:
                    tf[token] += 1

                # Normalize TF
                for token in tf:
                    tf[token] = 1 + math.log10(tf[token])

                # Update document frequency
                for token in tf:
                    doc_count[token] += 1

                # Store the TF vector
                tf_idf_vectors[filename] = tf

    # Calculate TF-IDF
    for filename in tf_idf_vectors:
        for token in tf_idf_vectors[filename]:
            df = doc_count[token]
            idf = math.log10(total_docs / df)
            tf_idf_vectors[filename][token] *= idf

    # Normalize the TF-IDF vectors
    for filename, tfidf in tf_idf_vectors.items():
        norm = math.sqrt(sum(weight ** 2 for weight in tfidf.values()))
        for token in tfidf:
            tfidf[token] /= norm

def getidf(token):
    # Returns the IDF of a token. 
    stemmed_token = stemmer.stem(token)
    df = doc_count.get(stemmed_token, 0)
    if df == 0:
        return -1
    return math.log10(len(tf_idf_vectors) / df)

def getweight(filename, token):
    # Returns the normalized TF-IDF weight of a token in a document. 
    stemmed_token = stemmer.stem(token)
    return tf_idf_vectors.get(filename, {}).get(stemmed_token, 0)

def query(qstring):
    # Returns the document with the highest cosine similarity score for the query. 
    query_tokens = preprocess_text(qstring)
    query_tf = defaultdict(int)

    # Calculate query term frequency
    for token in query_tokens:
        query_tf[token] += 1

    # Normalize query TF
    for token in query_tf:
        query_tf[token] = 1 + math.log10(query_tf[token])

    # Normalize the query vector
    query_norm = math.sqrt(sum(weight ** 2 for weight in query_tf.values()))
    for token in query_tf:
        query_tf[token] /= query_norm

    # Calculate cosine similarity and find the document with the highest cosine similarity
    best_score = float('-inf')
    best_doc = None

    for filename, doc_vector in tf_idf_vectors.items():
        score = sum(query_tf[token] * doc_vector.get(token, 0) for token in query_tf)
        if score > best_score:
            best_score = score
            best_doc = filename

    if best_doc is None:
        return ("None", 0)
    return (best_doc, best_score)

# Preprocess documents and compute TF-IDF
compute_tf_idf()

print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))

end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Print the execution time
print(f"Execution Time: {execution_time:.6f} seconds")
