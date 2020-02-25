# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:36:56 2020

@author: SOMESH
"""

import nltk
nltk.download()


paragraph ='''
Ahead of U.S. President Donald Trump’s visit to India, some of the key deliverables from the trip, as well as the outcomes that may not be delivered after his meeting with Prime Minister Narendra Modi on Tuesday, are coming into view. The larger question remains as to whether the bonhomie between the two, who will be meeting for the fifth time in eight months, will also spur the bilateral relationship towards broader outcomes, with expectations centred at bilateral strategic ties, trade and energy relations as well as cooperation on India’s regional environment. On the strategic front, India and the U.S. are expected to take forward military cooperation and defence purchases totalling about $3 billion. Mr. Trump has cast a cloud over the possibility of a trade deal being announced, but is expected to bring U.S. Trade Representative Robert Lighthizer to give a last push towards the trade package being discussed for nearly two years. Both sides have lowered expectations of any major deal coming through, given that differences remain over a range of tariffs from both sides; market access for U.S. products; and India’s demand that the U.S. restore its GSP (Generalised System of Preferences) status. However, it would be a setback if some sort of announcement on trade is not made. A failure to do so would denote the second missed opportunity since Commerce Minister Piyush Goyal’s U.S. visit last September. Finally, much of the attention will be taken by India’s regional fault-lines: the Indo-Pacific strategy to the east and Afghanistan’s future to the west. India and the U.S. are expected to upgrade their 2015 joint vision statement on the Indo-Pacific to increase their cooperation on freedom of navigation, particularly with a view to containing China. Meanwhile, the U.S.-Taliban deal is expected to be finalised next week, and the two leaders will discuss India’s role in Afghanistan, given Pakistan’s influence over any future dispensation that includes the Taliban.Any high-level visit, particularly that of a U.S. President to India, is as much about the optics as it is about the outcomes. It is clear that both sides see the joint public rally at Ahmedabad’s Motera Stadium as the centrepiece of the visit, where the leaders hope to attract about 1.25 lakh people in the audience. Despite the Foreign Ministry’s statement to the contrary, the narrative will be political. Mr. Trump will pitch the Motera event as part of his election campaign back home. By choosing Gujarat as the venue, Mr. Modi too is scoring some political points with his home State. As they stand together, the two leaders, who have both been criticised in the last few months for not following democratic norms domestically, will hope to answer their critics with the message that they represent the world’s oldest democracy and the world’s largest one, respectively.
'''
#tokenizing sentences
sentences = nltk.sent_tokenize(paragraph) 

#tokenizing words
words = nltk.word_tokenize(paragraph)

'''
Stemming
The process of reducing infected or derived words to their word stem,base or root form



'''
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()

for i in range(len(sentences)):
    words_stem = nltk.word_tokenize(sentences[i])
    words_stem = [stemmer.stem(word) for word in words_stem if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words_stem)
'''
Lemmatization
same as stemming but intermediate representation has meaning

'''

from nltk.stem import WordNetLemmatizer

sentences_lm = nltk.sent_tokenize(paragraph) 
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences_lm)):
    words_lem = nltk.word_tokenize(sentences_lm[i])
    words_lem = [lemmatizer.lemmatize(word) for word in words_lem if word not in set(stopwords.words('english'))]
    sentences_lm[i] = ' '.join(words_lem)

#Bag of words
#cleaning the texts
import re
#stemmer
#lemmatizer
corpus=[]
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

corpus_lem =[]    
for i in sentences_lm:
    review_lem = i.replace('U.S.','america')
    review_lem = re.sub('[^a-zA-Z]',' ',review_lem)
    review_lem = review_lem.lower()
    review_lem = review_lem.split()
    review_lem = [lemmatizer.lemmatize(word) for word in review_lem if not word in set(stopwords.words('english'))]
    review_lem = ' '.join(review_lem)
    corpus_lem.append(review_lem)
    
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()
#X = cv.fit_transform(corpus_lem).toarray()
#tf-idf term frequency and inverse document frequency
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus_lem).toarray() 

import pandas as pd
df = pd.DataFrame(corpus_lem)













































