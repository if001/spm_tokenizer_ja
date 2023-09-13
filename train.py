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

def train_aozora(tokenizer, trainer, dataset):
    def ds_yielder():
        for v in dataset["text"]:
            yield v
    tokenizer.train_from_iterator(ds_yielder(), trainer=trainer, length=len(dataset))
    return tokenizer

def train(tokenizer, trainer, dataset):
    def ds_yielder():
        for v in dataset:
            yield v["text"]
    tokenizer.train_from_iterator(ds_yielder(), trainer=trainer, length=len(dataset))
    return tokenizer
    
def split_array(arr, M):
    return [arr[i:i+M] for i in range(0, len(arr), M)]

# def train_with_split(tokenizer, trainer, dataset, batch_size, split_count=10):
#     base = range(len(dataset))
#     window = int(len(dataset)/split_count)
#     for i, idxs in enumerate(split_array(base, window)):                    
#         part = dataset.select(idxs)
#         print(part)
#         tokenizer = train_as_dataset(tokenizer, trainer, part , batch_size)
#     return tokenizer


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

    for dataset_id in args.datasets: 
        dataset = datasets.load_dataset(dataset_id)
        ds = dataset['train']
        print('raw dataset: ', ds)
        if "aozora" in dataset_id:
            tokenizer = train_aozora(tokenizer, trainer, ds)
        else:
            tokenizer = train(tokenizer, trainer, ds)

        save_id = dataset_id.split("/")[-1]
        save_file = f"./tmp_{save_id}.json"
        tokenizer.save(save_file)
        print(f'save... {save_file}')

    tokenizer.save(args.save_file)
    print(f'save... {args.save_file}')


if __name__ == '__main__':
    main()