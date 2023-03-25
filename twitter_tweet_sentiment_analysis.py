import tweepy
from textblob import TextBlob

# Twitter API credentials
consumer_key = "consumer_key"
consumer_secret ="consumer_secret"
access_token = "access_token"
access_token_secret = "access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

# Define search term and number of tweets to retrieve
search_term = "climate change"
num_tweets = 10000

# Retrieve tweets
tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang="en").items(num_tweets)

# Process tweets and perform sentiment analysis
positive_tweets = 0
negative_tweets = 0
neutral_tweets = 0

for tweet in tweets:
    # Preprocess tweet
    text = tweet.text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\@\w+", "", text)
    text = re.sub(r"\#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Perform sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    # Categorize tweet based on sentiment
    if sentiment > 0:
        positive_tweets += 1
    elif sentiment < 0:
        negative_tweets += 1
    else:
        neutral_tweets += 1

# Print results
print("Positive tweets: {}".format(positive_tweets))
print("Negative tweets: {}".format(negative_tweets))
print("Neutral tweets: {}".format(neutral_tweets))