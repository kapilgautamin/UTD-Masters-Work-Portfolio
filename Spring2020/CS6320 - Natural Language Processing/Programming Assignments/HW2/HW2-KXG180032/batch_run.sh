#!/bin/bash
time python sentiment.py train test bow regression 0 no > output.txt
time python sentiment.py train test bow regression 1 no >> output.txt
time python sentiment.py train test bow regression 0 l1 >> output.txt
time python sentiment.py train test bow regression 1 l1 >> output.txt
time python sentiment.py train test bow regression 0 l2 >> output.txt
time python sentiment.py train test bow regression 1 l2 >> output.txt

time python sentiment.py train test tfidf regression 0 no >> output.txt
time python sentiment.py train test tfidf regression 1 no >> output.txt
time python sentiment.py train test tfidf regression 0 l1 >> output.txt
time python sentiment.py train test tfidf regression 1 l1 >> output.txt
time python sentiment.py train test tfidf regression 0 l2 >> output.txt
time python sentiment.py train test tfidf regression 1 l2 >> output.txt

time python sentiment.py train test bow nbayes 0 >> output.txt
time python sentiment.py train test bow nbayes 1 >> output.txt

time python sentiment.py train test tfidf nbayes 0 >> output.txt
time python sentiment.py train test tfidf nbayes 1 >> output.txt



