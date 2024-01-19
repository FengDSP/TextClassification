import pandas as pd
import numpy as np
import time
import torch
import sys
import os
from absl import app
from absl import flags
from absl import logging
from torch import nn
from collections import defaultdict
from multiprocessing import Pool


from transformers import BertTokenizer

random_input = False
TITLES_TO_CATEGORIES_CSV = './titles_to_categories.csv'
max_title_token_length = 50
sample_frac = 0.1

OOV_TOKEN_ID = 1

def get_input_tensor(df, seq_len, vocab):
    df = df.reset_index(drop=True)
    data_tensor = torch.zeros((len(df), seq_len), dtype=torch.long)
    for i, row in df.iterrows():
        title = row["tokenized_title"]
        title = title[:seq_len]
        padded_title = title + ["[PAD]"] * (seq_len - len(title))
        for j, token in enumerate(padded_title):
            data_tensor[i][j] = vocab.get(token, OOV_TOKEN_ID)
    return data_tensor

def process_chunk(sub_df, seq_len, vocab, debug_index=0):
    # logging.info(f"Process {debug_index} starting ..")
    sub_data_tensor = torch.zeros((len(sub_df), seq_len), dtype=torch.long)
    for i, row in enumerate(sub_df.itertuples(index=False)):
        title = row.tokenized_title[:seq_len]
        padded_title = title + ["[PAD]"] * (seq_len - len(title))
        for j, token in enumerate(padded_title):
            sub_data_tensor[i][j] = vocab.get(token, OOV_TOKEN_ID)
    # logging.info(f"Process {debug_index} finished")
    return sub_data_tensor

#def debug_process_chunk(sub_df_len, seq_len, vocab, debug_index):
#    logging.info(f"Process {debug_index} starting ..")
#    sub_data_tensor = torch.zeros((sub_df_len, seq_len), dtype=torch.long)
#    logging.info(f"Process {debug_index} finished")
#    return sub_data_tensor


def distributed_get_input_tensor(df, seq_len, vocab, num_cores=96):
    df = df.reset_index(drop=True)
    # Split DataFrame into chunks
    chunk_size = len(df) // num_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    logging.info("Starting pool ..")
    with Pool(num_cores) as pool:
        results = pool.starmap(process_chunk, [(chunk, seq_len, vocab, i) for i, chunk in enumerate(chunks)])
    del pool
    data_tensor = torch.cat(results, dim=0)

    return data_tensor


def get_label_tensor(df):
    return torch.tensor(df["category_label"].tolist())


def tokenize_titles(sub_df, tokenizer):
    sub_df["tokenized_title"] = sub_df["title"].apply(lambda title: tokenizer.tokenize(title))
    return sub_df
    

def parallel_tokenize(df, tokenizer, num_cores=96):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.starmap(tokenize_titles, [(sub_df, tokenizer) for sub_df in df_split]))
    pool.close()
    pool.join()
    return df


def load_data(input_csv, sample_frac=0.1):
    df = pd.read_csv(TITLES_TO_CATEGORIES_CSV)
    df = df.sample(frac=sample_frac).reset_index(drop=True)
    logging.info(f"Loaded examples {len(df)}")
    categories = df['category_name'].unique()
    logging.info(f'Number of categories: {len(categories)}')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer.tokenize("I have a new GPU!")
    logging.info("Running tokenizer ...")
    # df["tokenized_title"] = df["title"].apply(lambda title: tokenizer.tokenize(title))
    df = parallel_tokenize(df, tokenizer)
    logging.info("Tokenizer finished.")
    df["category_label"] = df["category_name"].apply(lambda category: categories.tolist().index(category))
    # df[df["category_name"] == "Game Hardware"].head()

    # split the dataset into train and test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    return train_df, test_df, categories

def get_vocab(df, min_count=10):
    vocab_counts = defaultdict(lambda: 0)
    for tokenized_title in df["tokenized_title"]:
        for token in tokenized_title:
            vocab_counts[token] += 1
    token_counts = [(v, i) for v, i in vocab_counts.items() if i >= min_count]
    tokens_by_count = sorted(token_counts, key=lambda x: x[1], reverse=True)
    tokens_by_count = [('[PAD]', 0), ('[OOV]', 0)] + tokens_by_count
    vocab = {token: i for i, (token, _) in enumerate(tokens_by_count)}
    logging.info(f"Vocab items {len(vocab)}")
    return vocab


class MLPModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, seq_length, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        hidden_layers = []
        for fan_in_dim, fan_out_dim in zip([embed_dim * seq_length] + hidden_dims[:-1], hidden_dims):
            hidden_layers.extend([
                nn.Linear(fan_in_dim, fan_out_dim),
                nn.ReLU(),
            ])
        self.hidden_layers = nn.ModuleList(hidden_layers)
        fc_fan_in_dim = hidden_dims[-1] if hidden_dims else embed_dim
        self.fc = nn.Linear(fc_fan_in_dim, num_classes)

    def forward(self, x):
        # logging.info(f"input shape={x.shape}")
        x = self.embedding(x)
        # logging.info(f"embedding shape={x.shape}")
        x = torch.reshape(x, x.shape[:-2] + (-1,))
        # logging.info(f"concated shape={x.shape}")
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        # logging.info(f"last layer shape={x.shape}")
        x = self.fc(x)
        # logging.info(f"output shape={x.shape}")
        return x


def train(
    model,
    input_tensor,
    label_tensor,
    optimizer,
    batch_size=256,
    epochs=1,
    steps=None,
    logger=logging.info,
):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for i in range(0, input_tensor.shape[0], batch_size):
            batch_input_tensor = input_tensor[i:i+batch_size]
            # logging.info(f"batch_input_tensor.shape={batch_input_tensor.shape}")
            batch_label_tensor = label_tensor[i:i+batch_size]
            # logging.info(f"batch_label_tensor.shape={batch_label_tensor.shape}")
            optimizer.zero_grad()
            output = model(batch_input_tensor)
            # logging.info(f"output.shape={output.shape}")
            loss = loss_fn(output, batch_label_tensor).mean()
            # logging.info(f"loss.shape={loss.shape}")
            loss.backward()
            optimizer.step()
            # break
            if i / batch_size % 100 == 0:
                logging.info(f"Epoch {epoch} step {i / batch_size} loss: {loss.item()}")
            if steps and i >= steps:
                    break
        end_time = time.time()
        logger(f"Epoch {epoch} loss: {loss.item()}  time: {(end_time - start_time):.2f}s")
        # break
    return


def eval(model, input_tensor, label_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(label_tensor.view_as(pred)).type(torch.float).mean()
    logging.info(f"Accuracy: {correct.item() * 100:.2f}%")


def create_mlp_e32_512(vocab, categories):
    return MLPModel(
        vocab_size=len(vocab),
        embed_dim=32,
        hidden_dims=[512],
        seq_length=max_title_token_length,
        num_classes=len(categories),
    )


def ddp_main_fn(
    rank,
    world_size,
    vocab,
    categories,
    train_input_tensor,
    train_label_tensor,
):
    def loginfo(msg):
        sys.stdout.write(f"P{rank}] {msg}\n")

    loginfo(f"Starting ddp_main_fn rank {rank} of world_size {world_size} ...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '18121'
    
    torch.distributed.init_process_group('gloo', rank=rank, world_size=world_size)

    model = create_mlp_e32_512(vocab, categories).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ddp_optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    train(
        ddp_model,
        train_input_tensor.to(rank),
        train_label_tensor.to(rank),
        ddp_optimizer,
        batch_size=1024,
        logger=loginfo,
    )
    # if rank == 0:
        


def main(_):
    if not random_input:
        logging.info("----------- Load data -----------")
        train_df, test_df, categories = load_data(TITLES_TO_CATEGORIES_CSV, sample_frac)
        vocab = get_vocab(train_df)
        logging.info("Preparing train input tensor ...")
        train_input_tensor = distributed_get_input_tensor(train_df, max_title_token_length, vocab)
        logging.info("Preparing train label tensor ...")
        train_label_tensor = get_label_tensor(train_df)
        logging.info("Preparing test input tensor ...")
        test_input_tensor = get_input_tensor(test_df, max_title_token_length, vocab,)
        logging.info("Preparing test label tensor ...")
        test_label_tensor = get_label_tensor(test_df)
    else:
        logging.info("----------- Generate Random Data for Debugging -----------")
        categories = range(100)
        vocab = range(10000)
        train_input_tensor = torch.randint(
            low=0, high=len(vocab), size=(500000, max_title_token_length), dtype=torch.long)
        train_label_tensor = torch.randint(
            low=0, high=len(categories), size=(train_input_tensor.shape[0], ), dtype=torch.long)
        test_input_tensor = torch.randint(
            low=0, high=len(vocab), size=(100000, max_title_token_length), dtype=torch.long)
        test_label_tensor = torch.randint(
            low=0, high=len(categories), size=(test_input_tensor.shape[0], ), dtype=torch.long)

    if False:
        logging.info("----------- CPU training -----------")
        mlp_e32_512 = create_mlp_e32_512(vocab, categories)
        mlp_e32_512_param_count = sum([p.numel() for p in mlp_e32_512.parameters()])
        logging.info(f"Total number of parameters for model mlp_e32_512: {mlp_e32_512_param_count}")
        optimizer_mlp_e32_512 = torch.optim.Adam(mlp_e32_512.parameters(), lr=0.001)
        train(
            mlp_e32_512,
            train_input_tensor,
            train_label_tensor,
            optimizer_mlp_e32_512,
            batch_size=1024,
        )
        logging.info("Evaluating on train set:")
        eval(mlp_e32_512, train_input_tensor, train_label_tensor)
    
        logging.info("Evaluating on test set:")
        eval(mlp_e32_512, test_input_tensor, test_label_tensor)

    if True:
        logging.info("----------- Single GPU training -----------")
        assert torch.cuda.is_available()
        device_train_labels = train_label_tensor.to('cuda:0')
        device_train_inputs = train_input_tensor.to('cuda:0')
        device_mlp_e32_512 = create_mlp_e32_512(vocab, categories).to('cuda:0')
        device_optimizer_mlp_e32_512 = torch.optim.Adam(device_mlp_e32_512.parameters(), lr=0.001)
        train(
            device_mlp_e32_512,
            device_train_inputs,
            device_train_labels,
            device_optimizer_mlp_e32_512,
            batch_size=1024,
        )

    if False:
        logging.info("----------- Data Parallel ------------")
        logging.info(f"GPU count = {torch.cuda.device_count()}")
        dd_mlp_e32_512 = nn.DataParallel(create_mlp_e32_512(vocab, categories))
        dd_mlp_e32_512 = dd_mlp_e32_512.to('cuda:0')
        dd_optimizer_mlp_e32_512 = torch.optim.Adam(dd_mlp_e32_512.parameters(), lr=0.001)
        device_train_labels = train_label_tensor.to('cuda:0')
        device_train_inputs = train_input_tensor.to('cuda:0')
        train(
            dd_mlp_e32_512,
            device_train_inputs,
            device_train_labels,
            dd_optimizer_mlp_e32_512,
            batch_size=1024,
        )

    if True:
        logging.info("----------- Distributed Data Parallel ------------")
        world_size = 4
        assert world_size <= torch.cuda.device_count()
        torch.multiprocessing.spawn(
            ddp_main_fn,
            args=(world_size, vocab, categories, train_input_tensor, train_label_tensor),
            nprocs=world_size,
            join=True,
        )
        torch.distributed.destroy_process_group()

    logging.info("----------- Tensor Parallel ------------")
    
    logging.info("----------- Done -----------")


if __name__ == "__main__":
    app.run(main)
    
