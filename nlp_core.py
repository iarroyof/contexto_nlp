from twitter import *
from imdb import IMDb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import sys
sys.path.append(".")
import config


class profile_dicts(object):
    def __init__(self, sources, ctxt_user, access_key, access_secret,
                    consumer_key, consumer_secret, max_tweets=100,
                                                            n_topics=5):
        self.apis = {
            'twitter': Twitter(auth = OAuth(access_key,
                        access_secret,
                        consumer_key,
                        consumer_secret)),
            'imdb': IMDb()

                }
        # TODO: add option for model experimentation
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                                stop_words='english')
        self.model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                            learning_method='online',
                                            learning_offset=50.,
                                            random_state=0)
        self.max_tweets = max_tweets
        self.sources = sources
        #self.vectorizer = TfidfVectorizer()

    def filter_sources(self, unwanted_topics):
        # TODO: Remove sources/topics from
        # 'self.apis'/'self.models[i]['f_names']' according to 'unwanted_topics'
        return self

    def get_topics(self, model, interest, feature_names, no_top_words):
        # TODO: add word weights to the interest dictionary
        ids = []
        words = []
        for topic_idx, topic in enumerate(model.components_):
            ids.append(topic_idx)
            words.append([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

        return pd.DataFrame({'interest': [interest] * len(ids), 'id': ids,
                                                        'topic_words': words})

    def display_topics(self, model, interest, feature_names, no_top_words):
        print ("Interest %s:" % interest)
        for topic_idx, topic in enumerate(model.components_):
            print ("Topic %d:" % (topic_idx))
            print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def _fit_imdb(self, interest):
        srs = self.imdb.search_movie(interest)[0]
        # If the current interest movie has no synopsis
        # then continue to the next.
        mid = srs.movieID
        movie = self.imdb.get_movie(mid)
        try:
            self.synopses[interest]['synopsis'] = movie['synopsis']
        except KeyError:
            self.synopses[interest]['synopsis'] = None

        self.synopses[interest]['imdb_id'] = mid
        try:
            self.synopses[interest]['keywords'] = movie['keywords']
        except KeyError:
            self.synopses[interest]['keywords'] = None

        return self

    def _fit_twitter(self, interest):
        try:
            srs = self.twitter.search.tweets(q=interest)
        except TwitterHTTPError:
            return self

        for t, r in enumerate(srs):
            try:
                txt = r["text"]
            except KeyError:
                continue

            if txt in ['', None]:
                continue

            self.tweets[interest]["text"].append(txt)
            if t == self.max_tweets:
                break

        if self.tweets[interest]["text"] == []:
            self.tweets[interest]["text"] = None

        self.tweets[interest]["screen_name"] = srs["user"]["screen_name"]
        self.tweets[interest]["user_id"] = srs["user"]

        return self

    def fit(self, interests):
        self.imdb = self.apis['imdb']
        self.twitter = self.apis['twitter']

        self.synopses = {i: {'synopsis': '', 'imdb_id': 0, 'keywords': []}
                                                            for i in interests}
        self.tweets = {i: {'text': [], 'user_id': 0, 'screen_name': ''}
                                                            for i in interests}
        self.models = {i: {'model': None, 'f_names': []} for i in interests}
        # Information retrival and model fitting for each user's interest using
        # available APIs (sources).
        self.unavailable = []
        for i in interests:
            self._fit_imdb(i)
            self._fit_twitter(i)
            self.documents = self.tweets[i]['text'] + self.synopses[i]['synopsis']
            if self.documents == []:
                self.unavailable.append(i)
                continue

            tf = self.tf_vectorizer.fit_transform(self.documents)
            self.models[i]['f_names'] = self.tf_vectorizer.get_feature_names()
            self.models[i]['model'] = self.model.fit(tf)

        for i in interests:
            if i in self.unavailable:
                del self.models[i]

        return self

    def get_dict_interests(self, n_top_words=5, mode='get'):
        for i in list(self.models.keys()):
            if mode == 'display':
                self.display_topics(self.models[i]['model'], interest=i,
                                    feature_names=self.models[i]['f_names'],
                                    no_top_words=n_top_words)
            if mode == 'get':
                self.get_topics(self.models[i]['model'], interest=i,
                                feature_names=self.models[i]['f_names'],
                                                no_top_words=n_top_words)
