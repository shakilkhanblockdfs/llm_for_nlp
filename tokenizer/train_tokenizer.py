import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data/train.en,data/train.de',
    model_prefix='spm',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

