# Winograd-Schema-Challenge
A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences and requires the use of world knowledge and reasoning for its resolution. The schema takes its name from a well-known example by Terry Winograd

## Solution 
Here I use some method from those paper
- [Commonsense Knowledge Enhanced Embeddings for Solving Pronoun Disambiguation Problems in Winograd Schema Challenge](https://pdfs.semanticscholar.org/120a/ae102e17be4f1cc2b3c69a84229c60de8b9d.pdf)
- [Resolving Complex Cases of Definite Pronouns: The Winograd Schema Challenge](https://dl.acm.org/citation.cfm?id=2391032)

## NLP tools and commensense knowledge
- [spaCy -- Industrial-Strength Natural Language Processing](https://spacy.io/)
- [ConceptNet](http://conceptnet.io/)

## How can you use this code
I will use python language, here you just need install spacy and run the model
1. Install spaCy and download spacy model <br /> pip install -U spacy <br /> python -m spacy download en_core_web_sm

2. Just run python code without instructions <br />./NLP_PDPs.py <br />wait for it ends
