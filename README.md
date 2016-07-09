# WhoWroteMe
WhoWroteMe is an experimental tool who aims to solve authorship attribution problems.

It enables the user to create learning models based on different machine learning algorythms and lets the user choose which
to use for classification or even decide by a vote between them.

Currently 4 text features are used in model training:
    - Average sentance length
    - Most commot word length
    - Unusual words in content percent
    - Lexical diversity

# Used Libraries and Resourses
- [NLTK](http://www.nltk.org/) v3.2.1 is used for the lexicographical analysis of texts and generation of statistics for model creation.
- Models are created with Scikit-Learn v0.17.1.
- Text resources are from Project Gutenberg.
