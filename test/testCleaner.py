import pandas as pd  #pandas is used to create dataframes, which makes manipulations easy
import string  # string is imorted to remove the punctuations and digits
import re  #regular expressions
from textblob import TextBlob #textblob is a python library used to assign polarity and subjectivity to tweets
import nltk #nltk is a Natural Language Processing Library 
import json
import string
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

tweets_data_path = "testTweets.txt"  


def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

", ".join(stopwords.words('english'))

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

freq = ['rt', 'dm', 'hi']
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in freq])

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_analysis(score):
    if score >= 0.05:
      return 'positive'
    elif score <= - 0.05:
      return 'negative'
    else:
      return 'neutral'

from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(df['clean_tweet'].values.astype('U'))
    return X

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, random_state=42)

from sklearn.cluster import KMeans
k = 10

from sklearn.manifold import TSNE

while True:

    tweets_data = []  
    tweets_file = open(tweets_data_path, "r")  
    for line in tweets_file:  
        try:  
            tweet = json.loads(line)  
            tweets_data.append(tweet)  
        except:  
            continue
    tweets = pd.DataFrame()
    tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
    tweets['username'] = list(map(lambda tweet: tweet['user']['screen_name'], tweets_data))
    tweets['timestamp'] = list(map(lambda tweet: tweet['created_at'], tweets_data))
    tweets['location'] = list(map(lambda tweet: tweet['user']['location'], tweets_data))
    tweets['likes'] = list(map(lambda tweet: tweet['user']['favourites_count'], tweets_data))
    df = tweets
    df['clean_tweet'] = df['text'].apply(lambda text: strip_all_entities(strip_links(text)))
    df['clean_tweet'] = df['clean_tweet'].str.lower()
    df['clean_tweet'] = df['clean_tweet'].apply(lambda text: remove_stopwords(text))
    df["clean_tweet"] = df["clean_tweet"].apply(lambda text: remove_freqwords(text))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda text: lemmatize_words(text))
    df[['negative', 'neutral', 'positive', 'compound']] = df['clean_tweet'].apply(lambda x:pd.Series(analyzer.polarity_scores(x)))
    df['analysis'] = df['compound'].apply(get_analysis)


    text = df['clean_tweet'].values
    X = vectorize(text, 2 ** 12)
    X.shape

    pca_result = pca.fit_transform(X.toarray())
    pca_result.shape

    df['pca_x'] = xs=pca_result[:,0]
    df['pca_y'] = pca_result[:,1]
    df['pca_z'] = pca_result[:,2]

    
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(pca_result)
    y = y_pred
    df['y'] = y_pred

    df['cluster'] = pd.Series(y, index=df.index)

    tsne = TSNE(verbose=1, perplexity=100, random_state=42)
    X_embedded = tsne.fit_transform(X.toarray())

    df['tsne_x'] = X_embedded[:,0]
    df['tsne_y'] = X_embedded[:,1]

    df.to_csv('testTwitterClusterData.csv')