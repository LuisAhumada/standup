from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import pandas as pd
import os
from gensim import models
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk

#----------------------------------------------------------------------------------------------------
# 1. Downloading Transcripts from YouTube Links
#----------------------------------------------------------------------------------------------------

# same urls as you had... we can make a text file maybe that has the urls
#links = ["https://www.youtube.com/watch?v=KijAPJXjg8c", "https://www.youtube.com/watch?v=Fv3fkcCrn6k", "https://youtu.be/jQHN1ipLPdY"]

transcripts = []
data = pd.read_csv("./data/Comedians Dataset - Comedians.csv")
data = data.dropna(subset=["Link"])
data = data.tail(1)
for i in data["Link"]:
    video_id = i.split("v=")
    if len(video_id) == 1:
        video_id = i.split("e/")

    # get transcripts
    transcripts.append(YouTubeTranscriptApi.get_transcript(video_id[1]))

#----------------------------------------------------------------------------------------------------
# 2. Read in Laugh Data
#----------------------------------------------------------------------------------------------------
file_dict = dict()

# you might need to change this path and get rid of "standup"
# I had one extra folder layer on my comp
for subdir, dirs, files in os.walk(os.getcwd() + "/Results/"):
    for file in files:
        file_name = file.split(".")[0].lower()
        csv = pd.read_csv(subdir + file)
        file_dict[file_name] = csv

#----------------------------------------------------------------------------------------------------
# 3. Match Up Laugh Data to Text Data
#----------------------------------------------------------------------------------------------------

i = 0
for show in file_dict:
    previous_laugh_time = 0
    text_before_laugh = []
    for laugh_time in file_dict[show]['end']:
            text = ""
            for entry in transcripts[i]:
                time = entry['start'] + (entry['duration']/6)
                if time < laugh_time:
                   if time > previous_laugh_time:
                        text += " " + entry['text']
                else:
                    break
            text_before_laugh.append(text)
            previous_laugh_time = laugh_time
            print(text)
    file_dict[show]['joke'] = text_before_laugh
    i = i + 1
print(file_dict)
#----------------------------------------------------------------------------------------------------
# 4. Perform NLP Analysis on the Text
#----------------------------------------------------------------------------------------------------

# need a way to figure out subject
# maybe do LDA and LSA

###################################
# LDA
###################################
array_for_lda = []
for i in file_dict:
    for j in file_dict[i]["joke"]:
        array_for_lda.append(j)
print(array_for_lda)

nltk.download('wordnet')
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in array_for_lda]

dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(doc_term_matrix)
for i in doc_term_matrix:
    print(i)

# Creating the object for LDA model using gensim library
Lda = models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=6, num_words=10))
for topic in ldamodel.print_topics(num_topics=6, num_words=10):
    print(topic)
doc_topics = ldamodel[doc_term_matrix]
for i in doc_topics:
    print(i)

# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#5.-Build-the-Topic-Model
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=5,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = ldamodel.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

###################################
# LSA
###################################
# from sklearn.feature_extraction.text import CountVectorizer
#
# small_count_vectorizer = CountVectorizer(stop_words='english', max_features=400, ngram_range=(1,1))
# #small_text_sample = file_dict["chipotle"]#.sample(n=10000, random_state=0)#.as_matrix()
#
# #display(small_text_sample)
# print(file_dict["chipotle"]["joke"])
# small_document_term_matrix = small_count_vectorizer.fit_transform(file_dict["george"]["joke"])
# print(small_document_term_matrix.toarray())
# print(small_count_vectorizer.get_feature_names())
#
# #display(small_document_term_matrix)
#
# from sklearn.decomposition import TruncatedSVD
#
# n_topics = 8
#
# lsa_model = TruncatedSVD(n_components=n_topics)
# lsa_topic_matrix = lsa_model.fit_transform(small_document_term_matrix)
#
# from collections import Counter
# # Define helper functions
# def get_keys(topic_matrix):
#     '''returns an integer list of predicted topic categories for a given topic matrix'''
#     keys = []
#     for i in range(topic_matrix.shape[0]):
#         keys.append(topic_matrix[i].argmax())
#     return keys
#
# def keys_to_counts(keys):
#     '''returns a tuple of topic categories and their accompanying magnitudes for a given list of keys'''
#     count_pairs = Counter(keys).items()
#     categories = [pair[0] for pair in count_pairs]
#     counts = [pair[1] for pair in count_pairs]
#     return (categories, counts)
#
# lsa_keys = get_keys(lsa_topic_matrix)
# lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
# # Define helper functions
# def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
#     '''returns a list of n_topic strings, where each string contains the n most common
#         words in a predicted category, in order'''
#     top_word_indices = []
#     for topic in range(n_topics):
#         temp_vector_sum = 0
#         for i in range(len(keys)):
#             if keys[i] == topic:
#                 temp_vector_sum += document_term_matrix[i]
#         print(temp_vector_sum)
#         temp_vector_sum = temp_vector_sum.toarray()
#         top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
#         top_word_indices.append(top_n_word_indices)
#     top_words = []
#     for topic in top_word_indices:
#         topic_words = []
#         for index in topic:
#             temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
#             temp_word_vector[:,index] = 1
#             the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
#             topic_words.append(the_word.encode('ascii').decode('utf-8'))
#         top_words.append(" ".join(topic_words))
#     return top_words
#
# import numpy as np
# top_n_words_lsa = get_top_n_words(10, lsa_keys, small_document_term_matrix, small_count_vectorizer)
#
# for i in range(len(top_n_words_lsa)):
#     print("Topic {}: ".format(i), top_n_words_lsa[i])
#
#
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
#
# top_3_words = get_top_n_words(3, lsa_keys, small_document_term_matrix, small_count_vectorizer)
# labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]
#
# fig, ax = plt.subplots(figsize=(16,8))
# ax.bar(lsa_categories, lsa_counts)
# ax.set_xticks(lsa_categories)
# ax.set_xticklabels(labels)
# ax.set_title('LSA Topic Category Counts')
# plt.show()
