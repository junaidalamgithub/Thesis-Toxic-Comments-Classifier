## Developed by Md Junaid Alam as a part of MS Thessis

from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain

filename = 'toxic_comments_classifier.sav'
tf_idf_file = 'toxic_comments_tfidf_vectorizer.sav'
# load the saved model from disk
loaded_model = pickle.load(open(filename, 'rb'))


# Define the topic mappings
Toxic_mappings = { 0: "Toxic",
1: "Severe Toxic",
2: "Obscene",
3: "Threat",
4: "Insult",
5: "Identity Hate"  }

# load the saved tf_idf from disk
tf_idf = pickle.load(open(tf_idf_file, 'rb')) 

app = Flask(__name__)

# Default for base URL
@app.route("/")
def home():
    return "Rest Service for Toxic Comment Classification"

@app.route("/classify", methods=['POST'])
def classify():
    comment=[]
    comment.append(request.get_json()["comment"])
    vectorized_comnent = tf_idf.transform(comment)
    result = "The comment is tagged as: "
    result_array = loaded_model.predict(vectorized_comnent).toarray()
    isClean = True
    for i in range(len(result_array[0])):
        print(i)
        if(result_array[0][i]==1):
            result+="\n"+Toxic_mappings.get(i)
            isClean = False
    if(isClean):
          result = "The comment is tagged as clean"

    return  result

if __name__ == '__main__':
    app.run()