# contexto_nlp: Automated Q&amp;A for assessing lexicon acquisition

Use example:
```python
In [2]: from contexto_nlp.nlp_core import profile_dicts 
   ...: up = profile_dicts(sources=['imdb','twitter'], ctxt_user="user", access_
   ...: token="key", access_secret="pass", consumer_token="ckey", consumer_secre
   ...: t="cpass")                                                              

In [3]: up.fit(["lion king", "matrix"])                                         
Out[3]: <contexto_nlp.nlp_core.profile_dicts at 0x7f3cbc476358>

In [4]: up.get_dict_interests(n_top_words=5, mode='display')                    
Interest lion king:
Topic #0: lion king 2007 heiress american
Topic #1: simba scar pride rock mufasa
Topic #2: 2017 mini kingdom simba lion
Topic #3: american heiress 2007 simba scar
Topic #4: 1994 adventure vg rock pride
Interest matrix:
Topic #0: neo morpheus trinity smith matrix
Topic #1: 2010 raccord faux matrix marg
Topic #2: marg new job given guardian
Topic #3: smith mjolnir machine sentinels logos
```
