all: ns1 pm1 pm2 pm3

ns1:
	python neurosynth/0.neurosynth_setup.py

pm1:
	python pubmed_prep/1.get_articles.py

pm2:
	python pubmed_prep/1.2.prepare_articles.py

pm3:
	python pubmed_word2vec/2.pubmed_doc2vec.py
