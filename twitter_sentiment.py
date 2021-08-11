from textblob import TextBlob
import pandas as pd
tweet= pd.DataFrame()
def sentiment_analysis(df):
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    # Create a function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    # Create two new columns ‘Subjectivity’ & ‘Polarity’
    df['TextBlob_Subjectivity'] = df['tweet'].apply(getSubjectivity)
    df['TextBlob_Polarity'] =  df['tweet'].apply(getPolarity)
    df.loc[df.TextBlob_Polarity>0, 'sentiment']= 'positive'
    df.loc[df.TextBlob_Polarity == 0, 'sentiment'] = 'neutral'
    df.loc[df.TextBlob_Polarity<0, 'sentiment']= 'negative'
    return df





df= pd.DataFrame()
df['tweet']= pd.read_csv("E:/sentiment data2.csv",nrows=10, index_col=None).iloc[:,1]

sentiment_analysis(df)

df
