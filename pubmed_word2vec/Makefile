clean:
	rm -f doc2vec.o* bigram_transformer.pkl trigram_transformer.pkl doc_td.pkl cleaned_abstracts.pkl  model_check_results.pkl ns_abstracts_cleaned.pkl ns_inferred_vectors.npy

modelclean:
	rm -f doc2vec.o* models/doc2vec_trigram* model.txt 

dimtest:
	sbatch dimtest2.slurm

train:
	launch -s run_doc2vec.sh -r 48:00:00 -n doc2vec 

setup:
	launch -s setup.sh -n setup
