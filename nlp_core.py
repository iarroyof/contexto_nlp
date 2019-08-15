#from twitter import *
import twitter
from imdb import IMDb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd

import sys
sys.path.append(".")
import config


class profile_dicts(object):
    def __init__(self, sources, ctxt_user, access_token, access_secret,
                    consumer_token, consumer_secret, max_tweets=100, max_imdb=5,
                                                            n_topics=5):
        # TODO: get API parameters from json file
        self.apis = {
            'twitter': twitter.Api(consumer_key=[consumer_token],
                  consumer_secret=[consumer_secret],
                  access_token_key=[access_token],
                  access_token_secret=[access_secret]),
            #'twitter': Twitter(auth = OAuth(access_key,
            #            access_secret,
            #            consumer_key,
            #            consumer_secret)),
            'imdb': IMDb()

                }
        if sources != []:
            self.apis = {a: self.apis[a] for a in self.apis if a in sources}

        # TODO: add option for model experimentation
        self.max_tweets = max_tweets
        self.max_imdb = max_imdb
        self.sources = sources
        self.n_topics = n_topics
        self.analyser = SentimentIntensityAnalyzer()
        #self.vectorizer = TfidfVectorizer()

    def filter_sources(self, unwanted_topics):
        # TODO: Remove sources/topics from
        # 'self.apis'/'self.models[i]['f_names']' according to 'unwanted_topics'
        return self

    def get_topics(self, interest, n_top_words, sort_by='difficulty'):
        # TODO: add word weights to the interest dictionary
        ids = []
        topics = []
        feature_names = self.models[interest]['tfidf'].get_feature_names()
        for topic_idx, t in enumerate(self.models[interest]['model'] \
					  .components_):
            ids.append(topic_idx)
            topic = [(feature_names[i], self.models[interest]['tfidf'].idf_[i])
                                for i in t.argsort()[:-n_top_words - 1:-1]]
            if sort_by == 'difficulty':
                topic.sort(key=lambda x: x[1])
                topics.append(topic)

        return pd.DataFrame({'importance': ids, 'interest_topic': topics})

    def display_topics(self, model, n_top_words):

        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " | ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

    def _fit_imdb(self, interest):
        srs = self.imdb.search_movie(interest)
        srs = srs[:min(self.max_imdb, len(srs))]
        # If the current interest movie has no synopsis
        # then continue to the next.
        mids = [sr.movieID for sr in srs]

        for i in mids:
            movie = self.imdb.get_movie(i)
            try:
                synopsis = movie['synopsis'][0]
            except KeyError:
                try:
                    synopsis = movie['plot'][0]
                except KeyError:
                    try:
                        synopsis = movie['plot outline'][0]
                    except KeyError:
                        try:
                            synopsis = movie['long imdb title']
                        except:
                            continue

            self.synopses[interest]['synopses'].append(synopsis)
            self.synopses[interest]['imdb_ids'].append(i)
            try:
                self.synopses[interest]['titles'].append(movie['title'])
            except KeyError:
                self.synopses[interest]['titles'].append(None)

        return self

    def _fit_twitter(self, interest):
        try:
            srs = self.twitter.search.tweets(q=interest)
        except:
            return self

        for r in srs:
            try:
                txt = r["text"]
            except KeyError:
                continue

            if txt in ['', None]:
                continue

            self.tweets[interest]["texts"].append(txt)
            self.tweets[interest]["screen_names"].append(
                                                    srs["user"]["screen_name"])
            self.tweets[interest]["user_ids"].append(srs["user"])

        if self.tweets[interest]["texts"] == []:
           self.tweets[interest]["texts"] = None

        return self

    def fit(self, interests):
        self.imdb = self.apis['imdb']
        self.twitter = self.apis['twitter']

        self.synopses = {i: {'synopses': [], 'imdb_ids': [], 'titles': []}
                                                            for i in interests}
        self.tweets = {i: {'texts': [], 'user_ids': [], 'screen_names': []}
                                                            for i in interests}
        self.models = {i: {'model': None, 'f_names': []} for i in interests}
        self.documents = {}
        # Information retrival and model fitting for each user's interest using
        # available APIs (sources).
        self.unavailable = []
        for interest in interests:
            tf_vectorizer = TfidfVectorizer(stop_words='english')
            model = TruncatedSVD(n_components=self.n_topics)
            self._fit_imdb(interest)
            self._fit_twitter(interest)
            self.documents[interest] = self.tweets[interest]['texts'] \
                                + self.synopses[interest]['synopses']
            if self.documents[interest] == []:
                self.unavailable.append(interest)
                continue

            tf = tf_vectorizer.fit_transform(self.documents[interest])
            self.models[interest]['tfidf'] = tf_vectorizer #\
						   #.get_feature_names()
            self.models[interest]['model'] = model.fit(tf)

        return self

    def qa_fit(self, interest, n_top_words=5, n_top_sentences=4,
                    query_topic_len=3, sort_by='difficulty', query=None):
        assert query_topic_len < n_top_words
        preproces = self.models[interest]['tfidf'].build_preprocessor()
        tokenizer = self.models[interest]['tfidf'].build_tokenizer()
        idfs = self.models[interest]['tfidf'].idf_
        vocab = self.models[interest]['tfidf'].vocabulary_
        feature_names = self.models[interest]['tfidf'].get_feature_names()
        sentences = []
        for d in self.documents[interest]:
            sentences_str = nltk.sent_tokenize(d)
            for s in sentences_str:
                sentences.append(tokenizer(preproces(s)))

        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(s) for s in sentences]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
						num_topics=self.n_topics)
        index = similarities.SparseMatrixSimilarity(lsi[corpus],
                                                  num_features=len(dictionary))
        if query is None:
            tdf = self.get_topics(interest, sort_by=sort_by,
                                  n_top_words=n_top_words)
            topic_queries = [[w[0] for w in t]  for t in tdf.interest_topic]
            topic_results = []
            for q in topic_queries:
                q_bow = dictionary.doc2bow(q[:query_topic_len])
                q_tfidf = tfidf[q_bow]
                q_lsi = lsi[q_tfidf]
                index.num_best = n_top_sentences
                ranked_sent_ids = index[q_lsi]
                topic_results.append(
			[(" ".join(sentences[s]), r, len(sentences[s]))
                                               for s, r in ranked_sent_ids])

            plain_results = {'topic_rank': [], 'QA': [], 'length': [],
                             'sim_wrt_topic': [], 'ans': [],}
            for r, topic in enumerate(topic_results):
                for qu, sim, l in topic:
                    plain_results['topic_rank'].append(r)
                    plain_results['QA'].append(qu)
                    plain_results['length'].append(l)
                    plain_results['sim_wrt_topic'].append(sim)
                    ans = [(w, idfs[vocab[w]])
                              for w in tokenizer(preproces(qu))
                                                        if w in vocab]
                    ans.sort(key=lambda x: x[1])
                    plain_results['ans'].append(ans[:n_top_words])
            plain_results['sentiment'] = [
                self.analyser.polarity_scores(sentence)
                    for sentence in plain_results['QA']]
            return pd.DataFrame(plain_results)
        else:
            q = tokenizer(preproces(query))
            q_bow = dictionary.doc2bow(q[:query_topic_len])
            q_tfidf = tfidf[q_bow]
            q_lsi = lsi[q_tfidf]
            index.num_best = n_top_sentences
            ranked_sent_ids = index[query_tfidf]
            return [(sentences_str[s], r, len(sentences[s]))
                                               for s, r in ranked_sent_ids]

    def get_user_qa_plan(self, n_top_words=5, n_top_questions=4,
                         query_topic_len=3, sort_by='length'):
        plan = {}
        for i in list(self.models.keys()):
            plan[i] = self.qa_fit(interest=i, n_top_words=n_top_words,
                                  n_top_sentences=n_top_questions,
                                  query_topic_len=query_topic_len,
                                  sort_by='difficulty') \
                                .sort_values(by=[sort_by])
            plan[i].sort_values(by=[sort_by]) \
                    .to_csv('plain_results_' + '_'.join(i.split())  + '.csv',
                            index=False)
        return plan
