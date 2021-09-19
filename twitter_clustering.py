# -*- coding: utf-8 -*-
"""twitter_clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18UnWGqVmY19-2Q1tVnRu2CacsHCaoj60
"""

#----------------------- Data Scraping -----------------------#

#Import Required Modules
from tweepy.streaming import StreamListener #prints live tweets to console
from tweepy import OAuthHandler #Authenticates User APIs
from tweepy import Stream

# Twitter API keys are generted using twitter developer account. https://dev.twitter.com/apps/new, use this link to generate API keys
'''
Replaces the empty strings with API keys genrated, I removed them here as they are private
'''


# List of keywords that must be included in the tweets, which we will extract
hash_tags = ['Kabul', 'Afghanistan', 'Taliban']

# Initialize Global Count variable
# count = 0

# Input number of tweets to be downloaded
# num_tweets = 50

# Create the class that will handle the tweet stream.
class StdOutListener(StreamListener):
    '''
    This class is taken from tweepy documentation and a minor modification is made to download reuired number of tweets.
    '''
    def on_data(self, data):
        global count
        global num_tweets
        global stream
        # if count < num_tweets:
        file = open("tweets1.txt", "a")
        file.write(data)
        file.close()
            # count += 1
        return True
        # else:
        #     stream.disconnect()

    def on_error(self, status):
        print(status)


# Handles Twitter authentication
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

stream.filter(languages=["en"], track=hash_tags)  #stream.filter param is used to extract only desired tweets. In this program the params used are languages=['en'], this is used to extract only tweets in english language.

import pandas as pd  #pandas is used to create dataframes, which makes manipulations easy
import string  # string is imorted to remove the punctuations and digits
import re  #regular expressions
from textblob import TextBlob #textblob is a python library used to assign polarity and subjectivity to tweets
import nltk #nltk is a Natural Language Processing Library 
nltk.download('stopwords')
import json

tweets_data_path = "tweets1.txt"  
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

df.head()

# converting all tweets to lower case
import string

df['clean_tweet'] = df['clean_tweet'].str.lower()
df.head()

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
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df['clean_tweet'] = df['clean_tweet'].apply(lambda text: lemmatize_words(text))
df.head()

#!pip install vaderSentiment

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
print(df.head())

# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

hash_tags = hashtag_extract(df['text'])
hash_tags = list(filter(None, hash_tags)) 
hash_tags = [item for sublist in hash_tags for item in sublist]

import seaborn as sns
import matplotlib.pyplot as plt

# a = nltk.FreqDist(hash_tags)
# d = pd.DataFrame({'Hashtag': list(a.keys()),
#                   'Count': list(a.values())})
# # selecting top 10 most frequent hashtags     
# d = d.nlargest(columns="Count", n = 10) 
# plt.figure(figsize=(16,5))
# ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
# ax.set(ylabel = 'Count')
# plt.show()

# # Plotting and visualizing the counts
# plt.title('Sentiment Analysis')
# plt.xlabel('Sentiment')
# plt.ylabel('Counts')
# df['analysis'].value_counts().plot(kind = 'bar')
# plt.show()

# Let’s plot the results
# import matplotlib.pyplot as plt

# # time_subj = pd.Series(data=df['subjectivity'].values, index=df['timestamp'])
# # time_subj.plot(figsize=(16, 4), label="subjectivity", legend=True)
# # plt.show()

# time_polar = pd.Series(data=df['compound'].values, index=df['timestamp'])
# time_polar.plot(figsize=(16, 4), label="polarity", legend=True)
# plt.show()

# time_likes = pd.Series(data=df['likes'].values, index=df['timestamp'])
# time_likes.plot(figsize=(16, 4), label="Likes", legend=True)
# plt.show()

# df.plot.scatter(x='negative', y='positive')
# plt.show()

# df.plot.scatter(x='subjectivity', y='polarity')
# plt.show()

# all_words = ' '.join([text for text in df['clean_tweet'].astype(str)])
# from wordcloud import WordCloud
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text, maxx_features):
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(df['clean_tweet'].values.astype('U'))
    return X

text = df['clean_tweet'].values
X = vectorize(text, 2 ** 12)
X.shape

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
pca_result = pca.fit_transform(X.toarray())
pca_result.shape

df['pca_x'] = xs=pca_result[:,0]
df['pca_y'] = pca_result[:,1]
df['pca_z'] = pca_result[:,2]

from sklearn.cluster import KMeans

k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(pca_result)
y = y_pred
df['y'] = y_pred

df['cluster'] = pd.Series(y, index=df.index)
# print(df['cluster'])

# sns settings
# sns.set(rc={'figure.figsize':(10, 10)})

# # colors
# palette = sns.color_palette("bright", len(set(y)))

# # plot
# sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y, legend='full', palette=palette)
# plt.title("PCA Covid-19 Tweets - Clustered (K-Means) - Tf-idf with Plain Text")
# # plt.savefig("plots/pca_covid19_label_TFID.png")
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=pca_result[:,0], 
#     ys=pca_result[:,1], 
#     zs=pca_result[:,2], 
#     c=y, 
#     cmap='tab10'
# )
# ax.set_xlabel('pca-one')
# ax.set_ylabel('pca-two')
# ax.set_zlabel('pca-three')
# plt.title("PCA Covid-19 Tweets (3D) - Clustered (K-Means) - Tf-idf with Plain Text")
# # plt.savefig("plots/pca_covid19_label_TFID_3d.png")
# plt.show()

from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())

df['tsne_x'] = X_embedded[:,0]
df['tsne_y'] = X_embedded[:,1]

df.to_csv('twitter_cluster_data.csv')