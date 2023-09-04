import datasets
import argparse
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def init_tokenizer():
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.UnicodeScripts()
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    # tokenizer.decoder = decoders.ByteLevel()
    tokenizer.decoder = decoders.BPEDecoder()   
    return tokenizer


def train_as_dataset(tokenizer, trainer, dataset, batch_size):
    def batch_iterator():
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
    return tokenizer

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file",
        required=True,
        type=str,        
    )
    parser.add_argument(
        "--datasets",
        type=lambda s: s.split(","),
        default="izumi-lab/wikipedia-ja-20230720,izumi-lab/wikinews-en-20230728,if001/aozorabunko-clean-sin"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000
    )
    args = parser.parse_args()
    print(args.datasets)
    print(args.save_file)
    print(args.vocab_size)

    tokenizer = init_tokenizer()
    trainer = trainers.UnigramTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"],
        unk_token="<UNK>"
    )

    for dataset_id in args.datasets: 
        dataset = datasets.load_dataset(dataset_id)
        dataset = dataset['train']
        tokenizer = train_as_dataset(tokenizer, trainer, dataset, 1000)
        tokenizer.save(f"./tmp_{dataset_id}".json)
    tokenizer.save(args.save_file)


if __name__ == '__main__':
    main()