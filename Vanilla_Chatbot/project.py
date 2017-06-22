from nltk.corpus import state_union

lisa = state_union.fileids()
dataset = []
for ele in lisa:
    dataset.append(state_union.raw(ele))
for i in range(len(dataset)):
    dataset[i] = dataset[i].encode('utf-8')
import string
data = dataset
for i in range(len(data)):
    data[i] = data[i].translate(None,string.punctuation)
	data[i] = data[i].translate(None,"\n")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True,max_features=250,stop_words='english')
words = vectorizer.fit_transform(data)
X = words.toarray()

feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(X).flatten()[::-1]
n = 3
top_n = feature_array[tfidf_sorting][:n]

