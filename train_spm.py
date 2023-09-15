import sentencepiece as spm


spm.SentencePieceTrainer.Train(
    input='corpus.txt',
    model_prefix='sentencepiece',
    vocab_size=32000,
    pad_id=3,
    pad_piece='[PAD]',
    user_defined_symbols=['<BOS>', '<EOS>', '<PAD>', '<SEP>', '<MASK>'],
    model_type='unigram'
)

