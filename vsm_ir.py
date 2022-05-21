# command: python3 .\vsm_ir.py "create_index" ".\cfc-xml_corrected\"

import sys
import os
import xml.etree.ElementTree as ET
import math 
import json

# ---------------------------------------- NOTES ----------------------------------------
# TODO: ofer
# [ ] - Implement cleanWord - stop words, stemming, lower, strip... 
# [ ] - Include CITES or AUTHORS in any way? 

# ---------------------------------------- CONSTANTS ----------------------------------------
CREATE_INDEX       = "create_index"
QUESTION_ARGV      = "question"
JSON_PATH          = "vsm_inverted_index.json"

RECORD             = "RECORD"
RECORDNUM          = "RECORDNUM"
TITLE              = "TITLE"
TOPIC              = "TOPIC"
ABSTRACT           = "ABSTRACT"
EXTRACT            = "EXTRACT"

# ----------------------------------------- GLOBALS ----------------------------------------

# docs_dict - keyword: record_num, value: { MAX_FREQ, WORD_COUNT_IN_DOC, WORDS_IN_DOC = { keyword: WORD, value: TF-IDF } }
docs_dict  = {} 
MAX_FREQ           = "max_freq"
WORD_COUNT_IN_DOC  = "word_count_in_doc"
WORDS_IN_DOC       = "words_in_doc"
TF_IDF             = "tf-idf"

# words_dict - keyword: word, value: { IDF, DOC_CONTAIN_WORD = { keyword: record_num, value: { COUNT_WORD_IN_DOC, TF } } }
words_dict = {} 
IDF                = "idf"
DOC_CONTAIN_WORD   = "doc_contain_word"
COUNT_WORD_IN_DOC  = "count_word_in_doc"
TF                 = "tf"

# ----------------------------------------- FUNCTIONS ----------------------------------------

def saveToJSON(): 
    index = {"words_dict" : words_dict, "docs_dict" : docs_dict}
    with open(JSON_PATH, 'w') as file:
        json.dump(index, file, indent=4)
        

def calc_IDF_And_TFIDF_Values():
    N = len( docs_dict.keys() )
    # print(f"calc_IDF_And_TFIDF_Values()\t N: {N}\n") # TODO: debug

    for word in words_dict.keys():
        # print(f"word: {word}\n") # TODO: debug

        # Calculate IDF for each word
        NT = len( words_dict[ word ][ DOC_CONTAIN_WORD ] )
        # print(f"NT: {NT}\n") # TODO: debug

        idf = math.log2( N / NT )
        # print(f"idf: {idf}\n") # TODO: debug
        words_dict[ word ][ IDF ] = idf

        # Calculate tf * idf for each word and each doc
        for record_num in docs_dict.keys():
            if record_num in words_dict[ word ][ DOC_CONTAIN_WORD ]:
                tf = words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ TF ]
                docs_dict[ record_num ][ WORDS_IN_DOC ][ word ] = tf * idf
                # print(f"tf-idf: {tf * idf}\n") # TODO: debug


# The function normalize the value of tf by the max_freq 
def calcTFValues( record_num ):
    # print(f"calcTFValues()\t record_num: {record_num}\n") # TODO: debug
    for word in docs_dict[ record_num ][ WORDS_IN_DOC ]:
        tf = words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ] / docs_dict[ record_num ][ MAX_FREQ ]
        words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ TF ] = tf
        # print(f"word: {word}\t tf: {tf}\n") # TODO: debug


def insertToWordsDict( word, record_num ):
    # Word is -1 if word isn't valid (contain in Stop words)
    if ( word == -1 ):
        # print(f"insertToWordsDict()\t word in '-1'") # TODO: debug
        return 
    
    # Check if word is found in words dict - if not, insert new entity with empty DOC_CONTAIN_WORD sub-dict and IDF as zero
    if ( word not in words_dict.keys() ):
        # print(f"insertToWordsDict()\t word not in words_dict") # TODO: debug
        words_dict[ word ] = { IDF : 0, DOC_CONTAIN_WORD : {} }

    # Check if record_num is found in words_dict[ word ] - if not, insert new entity with COUNT_WORD_IN_DOC and TF as zeros
    if ( record_num not in words_dict[ word ][ DOC_CONTAIN_WORD ].keys() ):
        words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ] = { COUNT_WORD_IN_DOC : 0, TF : 0 }
    
    # Increment COUNT_WORD_IN_DOC
    words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ] += 1


def insertToDocsDict( word, record_num ):
    # Word is -1 if word isn't valid (contain in Stop words)
    if ( word == -1 ):
        # print(f"insertToDocsDict()\t word in '-1'\n") # TODO: debug
        return 
    
    # Insert word to docs_dict with TF-IDF as aero
    docs_dict[ record_num ][ WORDS_IN_DOC ][ word ] = { TF_IDF : 0 }


def cleanWord( word ):
    cleanWord = word.strip().lower()
    
    # TODO: ofer 
    # remove punctuation, stemming, remove stopwords 
    # if word isn't valid return -1
    return cleanWord


def extractWords( record ):
    doc_words = []

    # Add title's words to dox_words 
    title_words = record.find( TITLE )
    if title_words != None:
        doc_words += title_words.text.split()

    # Add topic's words to dox_words 
    topic_words = record.find( TOPIC )
    if topic_words != None:
        doc_words += topic_words.text.split()

    # Add abstract's words to dox_words 
    abstract_words = record.find( ABSTRACT )
    if abstract_words != None:
        doc_words += abstract_words.text.split()

    # Add extract's words to dox_words 
    extract_words = record.find( EXTRACT )
    if extract_words != None:
        doc_words += extract_words.text.split()

    return doc_words


def parseFile( fullpath ):
    doc = ET.parse( fullpath )
    root = doc.getroot()
    records = root.findall( RECORD )
    for record in records:
        record_num = record.find( RECORDNUM ).text.strip().lstrip('0')
        # print(f"record_num: {record_num}\n") # TODO: debug
        
        # Insert doc to docs_dict with empty sub-dict of words_in_dict and word_count and max_freq as zeros 
        docs_dict[ record_num ] = { MAX_FREQ : 0, WORD_COUNT_IN_DOC : 0, WORDS_IN_DOC : {} } 

        # Extract all words from doc include: title, topic, abstract and extract
        doc_words = extractWords( record )

        max_freq = 0
        for word in doc_words:
            # Clean word using strip, lower, remove stop words, stemming etc. 
            
            # print(f"word: {word}") # TODO: debug
            word = cleanWord( word )
            # print(f"cleanWord: {word}") # TODO: debug

            # Insert word into words_dict and docs_dict
            insertToDocsDict(word, record_num)
            insertToWordsDict( word, record_num )

            # Update WORD_COUNT_IN_DOC, find word's freq and update max_freq if needed
            docs_dict[ record_num ][ WORD_COUNT_IN_DOC ] += 1
            word_freq = words_dict[ word ][ DOC_CONTAIN_WORD ][ record_num ][ COUNT_WORD_IN_DOC ]
            max_freq = max( max_freq, word_freq)
            
        # Update max_freq in doc and calculate TF values
        docs_dict[ record_num ][ MAX_FREQ ] = max_freq 
        calcTFValues( record_num )

        print(f"THE END!!\n\n") # TODO: debug


def createIndex( dir_path ):
    files = os.listdir( dir_path )
    for file in files:
        if ( file.endswith(".xml") ):
            fullpath = dir_path + file
            # print(f"fullpath: {fullpath}\n") # TODO: debug
            parseFile( fullpath )

    # Calculating 
    calc_IDF_And_TFIDF_Values()
    saveToJSON()



# TODO: Tom's implementation
def query():
    a = 0


if __name__ == '__main__':
    if ( sys.argv[1] == f"{CREATE_INDEX}" ):
        createIndex( sys.argv[2] )
    elif ( sys.argv[1] == f"{QUESTION_ARGV}" ):
        query()

