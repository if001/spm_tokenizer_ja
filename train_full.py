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


def train(tokenizer, trainer, dataset_ids):
    def ds_yielder():
        for dataset_id in dataset_ids: 
            print('start...', dataset_id)
            dataset = datasets.load_dataset(dataset_id)            
            ds = dataset['train']
            print('ds', ds)
            # ds = ds.select(range(0, 100))
            if 'aozora' in dataset_id:
                for v in ds["text"]:
                    yield v
            else:
                for v in ds:
                    yield v["text"]
    tokenizer.train_from_iterator(ds_yielder(), trainer=trainer)
    return tokenizer
    
def split_array(arr, M):
    return [arr[i:i+M] for i in range(0, len(arr), M)]


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
        # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"],
        unk_token="<UNK>"
    )
    tokenizer = train(tokenizer, trainer, args.datasets)
    tokenizer.save(args.save_file)
    print(f'save... {args.save_file}')


if __name__ == '__main__':
    main()