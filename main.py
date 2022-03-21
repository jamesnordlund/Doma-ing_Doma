from domadoma import corpus, modeling

if __name__ == '__main__':

    corpus.build_corpus(out_path='output')
    modeling.refine_model(out_path='output')
    modeling.make_word_cloud(out_path='output')
    modeling.word_prediction()
