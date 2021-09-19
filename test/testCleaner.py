import pandas as pd  #pandas is used to create dataframes, which makes manipulations easy
import string  # string is imorted to remove the punctuations and digits
import re  #regular expressions
from textblob import TextBlob #textblob is a python library used to assign polarity and subjectivity to tweets
import nltk #nltk is a Natural Language Processing Library 
import json

while True:

    tweets_data_path = "testTweets.txt"  
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

    tweets.head()

    df = tweets

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

    df['clean_tweet'] = df['text'].apply(lambda text: strip_all_entities(strip_links(text)))

    # converting all tweets to lower case
    import string

    df['clean_tweet'] = df['clean_tweet'].str.lower()

    from nltk.corpus import stopwords
    ", ".join(stopwords.words('english'))

    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    df['clean_tweet'] = df['clean_tweet'].apply(lambda text: remove_stopwords(text))
    df.head()

    freq = ['rt', 'dm', 'hi']
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in freq])

    df["clean_tweet"] = df["clean_tweet"].apply(lambda text: remove_freqwords(text))
    df.head()

    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    def lemmatize_words(text):
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

    df['clean_tweet'] = df['clean_tweet'].apply(lambda text: lemmatize_words(text))


    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df[['negative', 'neutral', 'positive', 'compound']] = df['clean_tweet'].apply(lambda x:pd.Series(analyzer.polarity_scores(x)))

    def get_analysis(score):
        if score >= 0.05:
            return 'positive'
        elif score <= - 0.05:
            return 'negative'
        else:
            return 'neutral'

    df['analysis'] = df['compound'].apply(get_analysis)
    df.to_csv('testAnalysis.csv')