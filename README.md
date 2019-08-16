# contexto_nlp: Automated Q&amp;A for assessing lexicon acquisition

Use example:
```python
In [2]: from contexto_nlp.nlp_core import profile_dicts 
   ...: 
   ...: up = profile_dicts(sources=["imdb", "twitter"], ctxt_user="user",
   ...: access_token="key", access_secret="pass", consumer_token="ckey",
   ...: consumer_secret="cpass")                                                              

In [3]: up.fit(["the lion king", "the matrix"])
Out[3]: <contexto_nlp.nlp_core.profile_dicts at 0x7f3cbc476358>

In [4]: plan = up.get_user_qa_plan()  # See CSV files...
In [5]: plan
Out[5]:
{'the lion king':                                                    QA  \
14                                  lion kingdom mini   
2                                   lion kingdom mini   
18                                  lion kingdom mini
...
                                              answers  length  \
14  [(lion, 1.0), (kingdom, 1.69314718056), (mini,...       3   
2   [(lion, 1.0), (kingdom, 1.69314718056), (mini,...       3   
18  [(lion, 1.0), (kingdom, 1.69314718056), (mini,...       3
...
                                            sentiment  sim_wrt_topic  \
14  {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...       0.992697   
2   {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...       0.992741   
18  {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound...       0.903997
...
    topic_rank  
14           3  
2            0  
18           4
... ,
'the matrix':                                                    QA  \
4                                   movieberto matrix   
5                                     they spar again   
1                         neo laughs but is unsettled
...
                                              answers  length  \
4        [(matrix, 1.0), (movieberto, 2.09861228867)]       2   
5                             [(spar, 2.09861228867)]       3   
1   [(neo, 1.40546510811), (laughs, 2.09861228867)...       5
...

```
