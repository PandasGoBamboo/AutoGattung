import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd 
import numpy as np
import pickle
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
from germalemma import GermaLemma
from datetime import datetime

########################## Funktionen

# POS-Tagging
def TagPOS_Text(text_list):
    pos_list = []
    for row in text_list:
        new_row = row.split(' ')
        tagged_sent = tagger.tag(new_row)
        pos_list.append(tagged_sent)
    return pos_list

# Lemmatisierung
def lemmatizeText(text_list):
    lemmatized = []
    for lists in text_list:
        lemmasOF = []
        for tuples in lists:
            try:
                lemma = lemmatizer.find_lemma(tuples[0], tuples[1])
                lemmasOF.append(lemma)
            except ValueError:
                lemmasOF.append(tuples[0])
                continue
        lemmatized.append(" ".join(lemmasOF))
    return lemmatized

# Tokenisierung
def tokenizeText(text_list):

    stripped = []
    stemmed = []

    # Initialisiert Snowballstemmer
    stemmer = SnowballStemmer('german')

    for sendung in text_list:

        sent_of_sendung = []
        #stem_of_sendung = []

        # Zerlegt jeden Text in Sätze
        tok_sentences = sent_tokenize(sendung, language='german')

        # Tokenisiert jedes Wort in jedem Satz
        for satz in tok_sentences:

            # Tokenisierung
            words = word_tokenize(satz)

            # Strip Satzzeichen
            new_words= [word for word in words if word.isalnum()]
            
            # Joined tokenisierte Wörter jedes Satzes einer Sendung
            sent_of_sendung.append(" ".join(new_words))

            # Stemming
            #stemmed_words = [stemmer.stem(word) for word in new_words]
            #stem_of_sendung.append(" ".join(stemmed_words))
            
        # Joined tokensierte Sätze einer Sendung
        stripped.append(" ".join(sent_of_sendung))

        # Joined gestemmte Sätze einer Sendung
        #stemmed.append(" ".join(stem_of_sendung))
   
    return [
        stripped #,
        #stemmed
    ]

# Counter für Dauer von Skriptausführung
startTime = datetime.now()

# Konsolenoutput
print('ich roedel....')

# Öffnet trainierten Korpus zum POS-Tagging    
with open('nltk_german_classifier_data.pkl', 'rb') as f:
    tagger = pickle.load(f)

# Einstellung um alle Dokumente zu mergen
data = pd.read_pickle('Filename.pkl')

# Ausführung des Codes für jeden Namen in der Liste
text = data['volltext']
titel = data['haupt_titel']
sonst_titel = data['sonst_titel']



# Tokenisierung 
transformed_text = tokenizeText(text)
transformed_titel = tokenizeText(titel)
transformed_sonst_titel = tokenizeText(sonst_titel)

 # Fügt Dataframe Spalten hinzu
data['stripped_titel'] = transformed_titel[0]
#data['stemmed_titel'] = transformed_titel[1]
data['stripped_sonst_titel'] = transformed_sonst_titel[0]
data['stripped_text'] = transformed_text[0]
#data['stemmed_text'] = transformed_text[1]

data.to_pickle('more_stripped_raw_data.pkl')
print('Tokenized')


############################ POS-Tagging

pos = TagPOS_Text(data['stripped_text'])

# Fügt Dataframe Spalten hinzu
#data['tagged'] = pos (ACHTUNG: LANGE SÄTZE SCHRÄNKEN DIE SPEICHERUNG EIN)

############################ Lemmatisierung
lemmatizer = GermaLemma()

lemmatized = lemmatizeText(pos)

# Fügt Dataframe Spalten hinzu
data['lemma'] = lemmatized

data.to_pickle('Filename.pkl')

print(datetime.now() - startTime)