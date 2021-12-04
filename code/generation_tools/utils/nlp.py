import pandas as pd
import enchant

d = enchant.Dict("en_US")
# install python package pattern
# noun
from pattern.en import referenced
from pattern.en import pluralize, singularize
# verb
from pattern.en import conjugate, lemma
from pattern.en import INFINITIVE, PRESENT, PAST, FUTURE, PARTICIPLE # tense
from pattern.en import PROGRESSIVE # ing
from pattern.en import SG, PL # singular plural

# Word Transform
def word_form_transform(df):
    # Please check the webpage first
    #     https://github.com/clips/pattern/wiki/pattern-en#conjugation
    
    # Noun
    noun = 'hour'
    print(noun)
    print(referenced(noun)) # an hour
    
    noun = 'child'
    print(noun)
    print(pluralize(noun)) # children
    
    # Verb
    verb = 'take'
    print(verb)
    print(conjugate(verb, tense = PAST, alias = '3sg') ) # 'took'
    
    # conjugate(verb, 
    #     tense = PAST,        # INFINITIVE, PRESENT, PAST, FUTURE
    #     alias = '3sg',       # '1sg', '2sg', '3sg', 'pl', 'part', '1sgp', '2sgp', '3gp', 'ppl', 'ppart', etc. 
    # )
    # conjugate('take', tense = PRESENT, alias = '3sg') # takes
    # conjugate('take', tense = PAST, alias = '3sg') # took
    # conjugate('take', tense = PRESENT, alias = '3sg', aspect = PROGRESSIVE ) # taking
    # conjugate('take', tense = PAST+PARTICIPLE, alias = '3sg') # taken
    # lexeme('take')# take, took, taking, taken


# Cleaning
import re
def  clean_text(df):
    df = df.str.lower()
    df = df.apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    # remove numbers
    df = df.apply(lambda elem: re.sub(r"\d+", "", elem))
    return df


# Stemming
import nltk
nltk.download('punkt')
def tokenize_text(df):
    from nltk.tokenize import sent_tokenize, word_tokenize
    df = df.apply(lambda x: word_tokenize(x))
    return df

from nltk.stem import SnowballStemmer
def word_stemmer(text):
    stem_text = [SnowballStemmer("english").stem(i) if not (i.endswith('y') or i.endswith('er') or i.endswith('e')) else i for i in text]
    return stem_text

def stem_text(df):
    df = df.apply(lambda x: word_stemmer(x))
    return df


# Lemmatization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i, pos='v') for i in text]
    return lem_text

def lemmatize_text(df):
    df = df.apply(lambda x: word_lemmatizer(x))
    return df


# Stop words
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
def filter_stopword(df):
    stop = stopwords.words('english')
    df = df.apply(lambda x: ' '.join([word for word in x if word not in stop])) # tokenized or untokenized x.split()
    return df


# POS tags
from nltk import word_tokenize, pos_tag, pos_tag_sents
nltk.download('averaged_perceptron_tagger')
    
def pos_filter(text):
#     tagged_texts = pos_tag_sents(text)
    selected_pos_tags = ['NN','NNS','VB','VBP','JJ']
    eliminated_words = ['person', 'people']
    selected_words = [word for (word, pos) in text if word not in eliminated_words and pos in selected_pos_tags]
    return ' '.join(selected_words)

def pos_filter_text(df):
    df = df.apply(lambda x: pos_filter(x))
    return df

## Processing
def process_text(df):
    df = clean_text(df)
    df = tokenize_text(df)
    # df = stem_text(df)
    df = lemmatize_text(df)
    # df = stem_text(df)
    df = filter_stopword(df)
    return df


## Only tokenizing
def process_text_until_tokenize(df):
    df = clean_text(df)
    df = filter_stopword(df)
    df = tokenize_text(df)
    return df

def _replace_somewhere(data_clean, mapping,verb_map,obj_map,action, video_id):
    verb_vocab = list(set([verb_map[x] for x in verb_map.keys()]))
    action_str= mapping[action[0]]
    verb_str = verb_map[action[0]]
    obj_str = obj_map[action[0]]
    verb_found = False
    obj_found = False
    verb_index = -1
    obj_index = -1
    end_index = -1
    clean_script = data_clean[data_clean.id == video_id]['clean_script']
    if len(clean_script.keys()) == 0:
        return action_str.replace(" from somewhere", "").replace(" somewhere", "")
    key = clean_script.keys()[0]
    clean_script = clean_script[key]
    for i, token in enumerate(clean_script):
        if not verb_found:
            if token == verb_str:
                verb_index = i
                verb_found = True
        elif not obj_found:
            if token == obj_str:
                obj_index = i
                obj_found = True
        else:
            if token in ['and', 'as', 'they', 'their', 'before', 'then', 'look',
                         'start', 'after', 'while', 'pick', 'when', 'theyre'] + verb_vocab:
                end_index = i
                break
            if token in ['person']:
                end_index = i - 1
                break
    if verb_index < obj_index and obj_index < end_index and obj_index - verb_index <= 4:
        action_str = ' '.join(clean_script[verb_index:end_index])
        if clean_script[end_index-1] in ['a', 'the']:
            action_str = mapping[action[0]].replace(" from somewhere", "").replace(" somewhere", "")
        if clean_script[end_index-1] in ['in', 'to', 'of']:
            action_str = action_str[:-2]
        if clean_script[end_index-1] in ['from']:
            action_str = action_str[:-4]
        #print(video_id)
        #print(action_str)
    else:
        action_str = action_str.replace(" from somewhere", "").replace(" somewhere", "")
    return action_str
