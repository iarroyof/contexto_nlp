#from twitter import *
import twitter
from imdb import IMDb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append(".")
import config


class profile_dicts(object):
    def __init__(self, sources, ctxt_user, access_token, access_secret,
                    consumer_token, consumer_secret, max_tweets=100, max_imdb=5,
                                                            n_topics=5):
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
        if not sources is None:
            self.apis = {a: self.apis[a] for a in self.apis if a in sources}

        # TODO: add option for model experimentation
        self.max_tweets = max_tweets
        self.max_imdb = max_imdb
        self.sources = sources
        self.n_topics = n_topics
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
                        for i in topic.argsort()[-no_top_words:]])

        return pd.DataFrame({'interest': [interest] * len(ids), 'id': ids,
                                                        'topic_words': words})

    def display_topics(self, model, interest, feature_names, no_top_words):
        print ("\nInterest %s:" % interest)
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " | ".join([feature_names[i]
                             for i in topic.argsort()[:-no_top_words - 1:-1]])
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

        for t, r in enumerate(srs):
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

        self.synopses = {i:{'synopses': [], 'imdb_ids': [], 'titles': []}
                                                            for i in interests}
        self.tweets = {i: {'texts': [], 'user_ids': [], 'screen_names': []}
                                                            for i in interests}
        self.models = {i: {'model': None, 'f_names': []} for i in interests}
        # Information retrival and model fitting for each user's interest using
        # available APIs (sources).
        self.unavailable = []
        for interest in interests:
            tf_vectorizer = TfidfVectorizer(stop_words='english')
            model = TruncatedSVD(n_components=self.n_topics)
            self._fit_imdb(interest)
            self._fit_twitter(interest)
            self.documents = self.tweets[interest]['texts'] \
                                + self.synopses[interest]['synopses']
            if self.documents == []:
                self.unavailable.append(interest)
                continue

            tf = tf_vectorizer.fit_transform(self.documents)
            self.models[interest]['f_names'] = tf_vectorizer\
						   .get_feature_names()
            self.models[interest]['model'] = model.fit(tf)

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
