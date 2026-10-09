"""English function words ignored by lexical matching (keyword search, attribution).

Deliberately small: it removes words that carry no topical information ("how", "the",
"does") while keeping everything that could be a domain term.
"""

_STOPWORD_TEXT = """
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each few
for from further had has have having he her here hers him his how i if in into is it
its itself just me more most my no nor not now of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up use used uses using very was we
were what when where which while who whom why will with would you your yours
"""

STOPWORDS = frozenset(_STOPWORD_TEXT.split())
