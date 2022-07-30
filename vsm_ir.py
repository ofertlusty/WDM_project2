# create_index command: python3 .\vsm_ir.py "create_index" ".\cfc-xml_corrected\"
# TODO: after all small details and questions are answered should check option of creating auxiliary functions for common functionality between tfidf and bm25.

import sys
import os
import xml.etree.ElementTree as ET
import math 
import json
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log, sqrt

# ---------------------------------------- CONSTANTS ----------------------------------------
CREATE_INDEX       = "create_index"
QUESTION_ARGV      = "query"
JSON_PATH          = "vsm_inverted_index.json"
ANSWER_PATH        = "ranked_query_docs.txt"
MAX_NUM_RESULTS    = 15     # TODO: need to find optimal value
TFIDF_THRESHOLD    = 0.0    # TODO: need to find optimal value
BM25_THRESHOLD     = 0.0    # TODO: need to find optimal value
K_PARAM            = 1.5    # TODO: need to find optimal value
B_PARAM            = 0.7   # TODO: need to find optimal value

RECORD             = "RECORD"
RECORDNUM          = "RECORDNUM"
TITLE              = "TITLE"
TOPIC              = "TOPIC"
ABSTRACT           = "ABSTRACT"
EXTRACT            = "EXTRACT"

# ----------------------------------------- GLOBALS ----------------------------------------

# docs_dict - struct: 
# keyword: record_num, value: { VECTOR_LENGTH, MAX_FREQ, WORD_COUNT_IN_DOC, WORDS_IN_DOC = { keyword: WORD, value: {"tf-idf" : TF-IDF } } }
docs_dict  = {} 
VECTOR_LENGTH      = "vector_length"
MAX_FREQ           = "max_freq"
WORD_COUNT_IN_DOC  = "word_count_in_doc"
WORDS_IN_DOC       = "words_in_doc"
TF_IDF             = "tf-idf"

# words_dict - struct:
# keyword: word, value: { IDF, DOC_CONTAIN_WORD = { keyword: record_num, value: { COUNT_WORD_IN_DOC, TF } } }
words_dict = {} 
IDF                = "idf"
DOC_CONTAIN_WORD   = "doc_contain_word"
COUNT_WORD_IN_DOC  = "count_word_in_doc"
TF                 = "tf"

# Set of words we want to invalidated for our inverted index as showen in class
stopWords = set( stopwords.words("english") )

# In use for stemming words
porterStemmer = PorterStemmer()

# ----------------------------------------- FUNCTIONS ----------------------------------------

# Save docsDict and wordsDict into json file
def saveToJSON(): 
    index = {"words_dict" : words_dict, "docs_dict" : docs_dict}
    with open(JSON_PATH, 'w') as file:
        json.dump(index, file, indent=4)    

# Calculate Sqrt vector's length and add it to the inverted index
def calcSqrtVectorLength():
    for record_num in docs_dict.keys():
        value = docs_dict[ record_num ][ VECTOR_LENGTH ]
        docs_dict[ record_num ][ VECTOR_LENGTH ] = math.sqrt(value)

# Calculate IDF and TF-IDF parameters and add it to the inverted index
def calc_IDF_And_TFIDF_Values():
    N = len( docs_dict.keys() )

    for word in words_dict.keys():
        # Calculate IDF for each word
        NT = len( words_dict[ word ][ DOC_CONTAIN_WORD ] )
        idf = math.log2( N / NT )
        words_dict[ word ][ IDF ] = idf

        # Calculate tf-idf for each word and each doc
        for record_num in docs_dict.keys():
            if record_num in words_dict[ word ][ DOC_CONTAIN_WORD ]:
                tf = words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ TF ]
                tf_idf = tf * idf
                docs_dict[ record_num ][ WORDS_IN_DOC ][ word ][ TF_IDF ] = tf_idf
                docs_dict[ record_num ][ VECTOR_LENGTH ] += tf_idf * tf_idf
    # TODO: the vector length probably needs += instead of = 


# The function normalize the value of tf by the max_freq 
def calcTFValues( record_num ):
    for word in docs_dict[ record_num ][ WORDS_IN_DOC ]:
        tf = words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ] / docs_dict[ record_num ][ MAX_FREQ ]
        words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ TF ] = tf

# Insert word and record_num into words_dict
def insertToWordsDict( word, record_num ):    
    # Check if word is found in words dict - if not, insert new entity with empty DOC_CONTAIN_WORD sub-dict and IDF as zero
    if ( word not in words_dict.keys() ):
        words_dict[ word ] = { IDF : 0, DOC_CONTAIN_WORD : {} }

    # Check if record_num is found in words_dict[ word ] - if not, insert new entity with COUNT_WORD_IN_DOC and TF as zeros
    if ( record_num not in words_dict[ word ][ DOC_CONTAIN_WORD ].keys() ):
        words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ] = { COUNT_WORD_IN_DOC : 0, TF : 0 }
    
    # Increment COUNT_WORD_IN_DOC
    words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ] += 1

# Insert word to docs_dict with TF-IDF as aero
def insertToDocsDict( word, record_num ):    
    docs_dict[ record_num ][ WORDS_IN_DOC ][ word ] = { TF_IDF : 0 }

# The function clean the word using lower, remove words contain in stopWords set or punctuation and use stemming.  
def parseWord( word ):
    cleanWord = porterStemmer.stem(word.translate(str.maketrans('', '', string.punctuation)).lower()) 
    if cleanWord in stopWords:
       return None
    else:
        return cleanWord

# The function extract all texts contains in the file and returns a joined string of all those texts combined.
# the words is being strips and lower case.
def extractWords( record ):
    doc_words = []

    # Add title's words to dox_words 
    title_words = record.find( TITLE )
    if title_words != None:
        doc_words += title_words.text.strip().split()

    # Add topic's words to dox_words 
    topic_words = record.find( TOPIC )
    if topic_words != None:
        doc_words += topic_words.text.strip().split()

    # Add abstract's words to dox_words 
    abstract_words = record.find( ABSTRACT )
    if abstract_words != None:
        doc_words += abstract_words.text.strip().split()

    # Add extract's words to dox_words 
    extract_words = record.find( EXTRACT )
    if extract_words != None:
        doc_words += extract_words.text.strip().split()

    return doc_words

# The function parses the input file, extracting the record num and the text contains in it. 
# Later on, the function goes word by word and cleans it. 
# Then, add the words into the inverted index while calculating the max frequency (for normalization), 
# the word count and all other parameters found in our inverted index (for details see the file attached).
def parseFile( fullpath ):
    doc = ET.parse( fullpath )
    root = doc.getroot()
    records = root.findall( RECORD )
    for record in records:
        record_num = record.find( RECORDNUM ).text.strip().lstrip('0')
        
        # Insert doc to docs_dict with empty sub-dict of words_in_dict and vector length, word_count and max_freq as zeros 
        docs_dict[ record_num ] = { VECTOR_LENGTH : 0, MAX_FREQ : 0, WORD_COUNT_IN_DOC : 0, WORDS_IN_DOC : {} } 

        # Extract all words from doc include: title, topic, abstract and extract
        doc_words = extractWords( record )

        max_freq = 0
        for word in doc_words:
            # Parse word using strip, lower, remove stop words and stemming etc
            cleanWord = parseWord( word )
            if (cleanWord is None):
                continue

            # Insert word into words_dict and docs_dict
            insertToDocsDict( cleanWord, record_num )
            insertToWordsDict( cleanWord, record_num )

            # Update WORD_COUNT_IN_DOC, find word's freq and update max_freq if needed
            docs_dict[ record_num ][ WORD_COUNT_IN_DOC ] += 1
            word_freq = words_dict[ cleanWord ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ]
            max_freq = max( max_freq, word_freq)
            
        # Update max_freq in doc and calculate TF values
        docs_dict[ record_num ][ MAX_FREQ ] = max_freq 
        calcTFValues( record_num )

# The function open every ".xml" file and parses into the inverted index (saved as a ".json" file).
def createIndex( dir_path ):
    files = os.listdir( dir_path )
    for file in files:
        if ( file.endswith(".xml") ):
            fullpath = dir_path + file
            parseFile( fullpath )

    # Calculating 
    calc_IDF_And_TFIDF_Values()
    calcSqrtVectorLength()

    # Save dictionaries to Json file
    saveToJSON()

    
def output_result(results: list):
    with open(ANSWER_PATH, 'w') as fp:
        for doc in results:
            fp.write("%s\n" % doc)


def tfidf_query(path: str, query: str):
    # TODO: must use the same stemming for query too! 
    # use parseWord()?
    inverted_index = json.load(open(path))
    q_words_dict = inverted_index['words_dict']     
    q_docs_dict = inverted_index['docs_dict']       
    clean_query = []                                # List containing all clean words in query
    query_dict = {}                                 # Dictionary with clean words as keys and tf-idf at values
    output_candidates = {}
    query_length = 0.0
    query_words = query.strip().split()
    for word in query_words:
        clean_word = parseWord(word)
        if clean_word is not None:
            clean_query.append(clean_word)
    for clean_word in clean_query:
        if clean_word in q_words_dict.keys():
            query_dict[clean_word] = query_dict.get(clean_word, 0) + 1
    # TODO: check if term frequency in query must be normalized by max frequency since in slide 76 in lecture notes it isn't
    max_frequency = max(query_dict.values())
    for term in query_dict:
        # calculate and updated tf * idf
        query_dict[term] = (query_dict[term] / max_frequency) * q_words_dict[term][IDF]
        for doc in q_words_dict[term][DOC_CONTAIN_WORD]:
            output_candidates[doc] = output_candidates.get(doc, 0.0) + query_dict[term] * q_docs_dict[doc][WORDS_IN_DOC][term][TF_IDF]
        query_length += query_dict[term] ** 2
    for candidate in output_candidates:
        output_candidates[candidate] /= q_docs_dict[candidate][VECTOR_LENGTH] * sqrt(query_length)

    # TODO: need to clean the finish of this method
    results = sorted(output_candidates.items(), key=lambda candidate: candidate[1], reverse=True)
    clean_results = [result[0] for result in results if result[1] > TFIDF_THRESHOLD]
    output_result(clean_results[:MAX_NUM_RESULTS])
    return


def bm25_query(path: str, query: str):
    # TODO: what happens when same word has several occurences in the query - posted question in forum.
    # TODO: the follow-up question is how to normalize the inner product <query, doc>
    # current implementataion ignores multiple occurrences
    inverted_index = json.load(open(path))
    q_words_dict = inverted_index['words_dict']     
    q_docs_dict = inverted_index['docs_dict']       
    clean_query = []                                # List containing all clean words in query
    query_dict = {}                                 # Dictionary with clean words as keys and tf-idf at values
    output_candidates = {}
    query_words = query.strip().split()
    N = len(q_docs_dict.keys())                       # Number of documents
    avgdl = 0.0
    for doc in q_docs_dict.keys():
        avgdl += q_docs_dict[doc][WORD_COUNT_IN_DOC]
    avgdl = avgdl / N
    for word in query_words:
        clean_word = parseWord(word)
        if clean_word is not None:
            clean_query.append(clean_word)
    for clean_word in clean_query:
        if clean_word in q_words_dict.keys():
            query_dict[clean_word] = query_dict.get(clean_word, 0) + 1
    for term in query_dict:
        count = len(q_words_dict[term][DOC_CONTAIN_WORD].keys())        # num of docs containing the term
        # calculate the bm25 idf according go slide 39 in lecture 4 slides
        idf = log( (N - count + 0.5) / (count + 0.5) + 1)
        for doc in q_words_dict[term][DOC_CONTAIN_WORD]:
            term_count = q_words_dict[term][DOC_CONTAIN_WORD][doc][COUNT_WORD_IN_DOC]
            bm25_weight = (term_count * (K_PARAM + 1)) / (term_count + K_PARAM * (1 - B_PARAM + B_PARAM * q_docs_dict[doc][WORD_COUNT_IN_DOC] / avgdl))
            output_candidates[doc] = output_candidates.get(doc, 0.0) + idf * bm25_weight
    results = sorted(output_candidates.items(), key=lambda candidate: candidate[1], reverse=True)
    clean_results = [result[0] for result in results if result[1] > BM25_THRESHOLD]
    output_result(clean_results[:MAX_NUM_RESULTS])
    return


def main(argv):
    if ( argv[1] == f"{CREATE_INDEX}" ):
        createIndex( argv[2] )
    elif ( argv[1] == f"{QUESTION_ARGV}" ):
        if argv[2] == 'tfidf':
            tfidf_query(argv[3], argv[4])
        elif argv[2] == 'bm25':
            bm25_query(argv[3], argv[4])

if __name__ == '__main__':
    main(sys.argv)
    """
    if ( sys.argv[1] == f"{CREATE_INDEX}" ):
        createIndex( sys.argv[2] )
    elif ( sys.argv[1] == f"{QUESTION_ARGV}" ):
        if sys.argv[2] == 'tfidf':
            tfidf_query(sys.argv[3], sys.argv[4])
        elif sys.argv[2] == 'bm25':
            bm25_query(sys.argv[3], sys.argv[4])
    """
