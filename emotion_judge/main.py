from textblob import TextBlob
import csv
from pathlib import Path

class DataManager:
    def load_tweet(self):
        data = []
        path = Path("./Tweets.csv")
        with path.open() as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        return data

class TextBlobHandler:
    # tweetがポジティブな内容か判定
    def is_positive(self, tweet: str):
        feedback_polarity = TextBlob(tweet).sentiment.polarity
        if feedback_polarity>0:
            return True
        return False
    

def write_csv(data, path: Path):
    with path.open(mode="w") as f:
        writer = csv.DictWriter(f, data[0].keys())
        writer.writeheader()
        for d in data:
            writer.writerow(d)
        
    
if __name__ == "__main__":
    data_manager = DataManager()
    text_blob_handler = TextBlobHandler()
    
    tweets = data_manager.load_tweet()
    
    # 感情分析の結果を保持
    result = []
    
    for tweet in tweets:
        if tweet["sentiment"] == "neutral":
            continue
        if text_blob_handler.is_positive(tweet["text"]):
            tweet["judge"] = "positive"
        else:
            tweet["judge"] = "negative"
        result.append(tweet)
    write_csv(result, Path("./result.csv"))
    # print(text_blob_handler.is_positive(tweets[0]["text"]))
    
    
    

# feedbacks = ['I love the app is amazing ', 
#              "The experience was bad as hell", 
#              "This app is really helpful",
#              "Damn the app tastes like shit ",
#             'Please don\'t download the app you will regret it ']

# positive_feedbacks = []
# negative_feedbacks = []

# for feedback in feedbacks:
#   feedback_polarity = TextBlob(feedback).sentiment.polarity
#   if feedback_polarity>0:
#     positive_feedbacks.append(feedback)
#     continue
#   negative_feedbacks.append(feedback)

# print('Positive_feebacks Count : {}'.format(len(positive_feedbacks)))
# print(positive_feedbacks)
# print('Negative_feedback Count : {}'.format(len(negative_feedbacks)))
# print(negative_feedbacks)

