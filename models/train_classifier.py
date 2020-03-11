import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import pickle
import re
import pandas as pd
import numpy as np
import sklearn
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score#, multilabel_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse',con=engine)
    X =df.message.values
    df_Y = df.loc[:,~df.columns.isin(['id','message','original','genre'])]
    df_Y['related']=df_Y['related'].map(lambda x:1 if x==2 else x)
    y=df_Y.values

    return X,y,df_Y



def tokenize(text):
    text = re.sub('[^a-zA-Z0-9_]',' ',text).lower().strip()
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")]
    clean_tokens = [WordNetLemmatizer().lemmatize(tok, pos='v') for tok in tokens]
  
    return clean_tokens




def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier())))])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
   


def evaluate_model(model, X_test, y_test, category_names):
    y_pred=model.predict(X_test)                       
    print('Accuracy=',accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred,target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()