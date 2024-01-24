import pandas as pd
import numpy as np
import datetime
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
        fc_fan_in_dim = hidden_dims[-1] if hidden_dims else embed_dim * seq_length
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


class CustomizedLinear(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input,  # ..., h1
        weights,  # h1, h2
        bias,  # h2
    ):
        ctx.save_for_backward(input, weights)
        return input @ weights + bias   # ..., h2

    @staticmethod
    def backward(
        ctx,
        grad_output,  # ..., h2
    ):
        input, weights = ctx.saved_tensors
        grad_input = grad_output @ weights.T
        grad_weights = input.reshape((-1, input.shape[-1])).T @ grad_output.reshape((-1, grad_output.shape[-1]))
        grad_bias = grad_output
        return grad_input, grad_weights, grad_bias


class MLPModelWithCustomizedLinear(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, seq_length, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        hidden_layer_weights = []
        hidden_layer_biases = []
        for fan_in_dim, fan_out_dim in zip([embed_dim * seq_length] + hidden_dims[:-1], hidden_dims):
            w = torch.empty(fan_in_dim, fan_out_dim)
            nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
            b = torch.zeros((fan_out_dim,))
            hidden_layer_weights.append(nn.parameter.Parameter(w))
            hidden_layer_biases.append(nn.parameter.Parameter(b))
        self.linear_layer_weights = nn.ParameterList(hidden_layer_weights)
        self.linear_layer_biases = nn.ParameterList(hidden_layer_biases)
        self.relus = nn.ModuleList([nn.ReLU()] * len(hidden_dims))
        fc_fan_in_dim = hidden_dims[-1] if hidden_dims else embed_dim * seq_length
        self.fc = nn.Linear(fc_fan_in_dim, num_classes)

    def forward(self, x):
        # logging.info(f"input shape={x.shape}")
        x = self.embedding(x)
        # logging.info(f"embedding shape={x.shape}")
        x = torch.reshape(x, x.shape[:-2] + (-1,))
        # logging.info(f"concated shape={x.shape}")
        for w, b, relu in zip(self.linear_layer_weights, self.linear_layer_biases, self.relus):
            x = CustomizedLinear.apply(x, w, b)
            x = relu(x)
        # logging.info(f"last layer shape={x.shape}")
        x = self.fc(x)
        # logging.info(f"output shape={x.shape}")
        return x


def loginfo(rank, msg):
    # sys.stdout.write(f"P{rank}] {msg}\n")
    pass


class TPMLPModelWithRowLinear(nn.Module):

    row_linear_comm = None
    fc_comm = None
    
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dims,
                 seq_length,
                 num_classes,
                 rank,
                 world_size,
                ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        assert embed_dim % world_size == 0
        embed_dim_per_rank = embed_dim // world_size
        self.embedding = nn.Embedding(vocab_size, embed_dim_per_rank, device=f"cuda:{rank}")
        hidden_layers = []
        for fan_in_dim, fan_out_dim in zip([embed_dim * seq_length] + hidden_dims[:-1], hidden_dims):
            assert fan_in_dim % world_size == 0
            fan_in_dim = fan_in_dim // world_size
            hidden_layers.extend([
                nn.Linear(fan_in_dim, fan_out_dim, bias=(rank == 0), device=f"cuda:{rank}"),
            ])
        self.relus = nn.ModuleList([nn.ReLU()] * len(hidden_layers))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        fc_fan_in_dim = hidden_dims[-1] // world_size if hidden_dims else embed_dim_per_rank * seq_length
        self.fc = nn.Linear(fc_fan_in_dim, num_classes, bias=(rank == 0), device=f"cuda:{rank}")
        
    def forward(self, x):
        loginfo(self.rank, f"input shape={x.shape}")
        x = self.embedding(x)
        loginfo(self.rank, f"embedding shape={x.shape}")
        x = torch.reshape(x, x.shape[:-2] + (-1,))
        loginfo(self.rank, f"concated shape={x.shape}")
        for hidden_layer, relu in zip(self.hidden_layers, self.relus):
            x = hidden_layer(x)
            loginfo(self.rank, f"hidden layer shape={x.shape}")
            # if not self.weight.requires_grad:
            #     self._forward_impl = linear_with_frozen_weight
            # else:
            #     self._forward_impl = linear_with_grad_accumulation
            x = self.__class__.row_linear_comm(x)
            x = relu(x)
            loginfo(self.rank, f"relu output shape={x.shape}") 
        loginfo(self.rank, f"last layer shape={x.shape}")
        x = self.fc(x)
        loginfo(self.rank, f"output shape={x.shape}")
        return self.__class__.fc_comm(x)


def direct_row_linear_comm(x):
    world_size = torch.distributed.get_world_size()
    # Prepare for the reduce scatter [B, H] -> [WorldSize, B, H/WorldSize]
    # this assumption is not necessary if we use torch.permute instead of torch.transpose
    assert len(x.shape) == 2
    x = torch.reshape(x, x.shape[:-1] + (world_size, x.shape[-1] // world_size))
    # loginfo(f"reshaped hidden layer shape={x.shape}")
    x = torch.transpose(x, 0, -2)
    # loginfo(f"transposed hidden layer shape={x.shape}")
    
    y = torch.zeros_like(x[0])
    # loginfo(f"reduce_scatter output shape={y.shape}")
    torch.distributed.reduce_scatter_tensor(y, x, op=torch.distributed.ReduceOp.SUM)
    return y


def direct_fc_comm(x):
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


class TPMLPModelWithDirectRowLinear(TPMLPModelWithRowLinear):
    row_linear_comm = direct_row_linear_comm
    fc_comm = direct_fc_comm


class RowLinearComm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input,  # b, h
    ):
        world_size = torch.distributed.get_world_size()
        # Prepare for the reduce scatter [B, H] -> [WorldSize, B, H/WorldSize]
        # this assumption is not necessary if we use torch.permute instead of torch.transpose
        assert len(input.shape) == 2
        x = torch.reshape(input, input.shape[:-1] + (world_size, input.shape[-1] // world_size))
        # loginfo(f"reshaped hidden layer shape={x.shape}")
        x = torch.transpose(x, 0, -2)
        # loginfo(f"transposed hidden layer shape={x.shape}")
        
        y = torch.zeros_like(x[0])
        # loginfo(f"reduce_scatter output shape={y.shape}")
        torch.distributed.reduce_scatter_tensor(y, x, op=torch.distributed.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(
        ctx,
        grad_output,  # b, h/world_size
    ):
        world_size = torch.distributed.get_world_size()
        all_grad_outputs = [torch.zeros_like(grad_output)] * world_size
        torch.distributed.all_gather(all_grad_outputs, grad_output)
        grad_input = torch.cat(all_grad_outputs, dim=-1)
        return grad_input


class AllReduceComm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        torch.distributed.all_reduce(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class TPMLPModelWithCustomizedRowLinear(TPMLPModelWithRowLinear):
    row_linear_comm = RowLinearComm.apply
    fc_comm = AllReduceComm.apply


class TPMLPModelWithColumnLinear(nn.Module):

    @staticmethod
    def column_linear_comm(x):
        # This doesn't work since backprop doesn't flow back through all_gather.
        # world_size = torch.distributed.get_world_size()
        # all_x = [torch.zeros_like(x, requires_grad=True)] * world_size
        # assert all_x[0].requires_grad
        # torch.distributed.all_gather(all_x, x)
        # x = torch.cat(all_x, dim=-1)
        # assert x.requires_grad
        # return x

        # to be implemented
        pass

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dims,
                 seq_length,
                 num_classes,
                 rank,
                 world_size,
                ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.num_classes = num_classes
        assert embed_dim % world_size == 0
        embed_dim_per_rank = embed_dim // world_size
        self.embedding = nn.Embedding(vocab_size, embed_dim_per_rank, device=f"cuda:{rank}")
        hidden_layers = []
        for fan_in_dim, fan_out_dim in zip([embed_dim * seq_length] + hidden_dims[:-1], hidden_dims):
            assert fan_out_dim % world_size == 0
            fan_out_dim_per_rank = fan_out_dim // world_size
            hidden_layers.extend([
                nn.Linear(fan_in_dim, fan_out_dim_per_rank, device=f"cuda:{rank}"),
            ])
        self.relus = nn.ModuleList([nn.ReLU()] * len(hidden_layers))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        fc_fan_in_dim = hidden_dims[-1] if hidden_dims else embed_dim * seq_length
        fc_fan_out_dim = (num_classes - 1) // world_size + 1
        self.fc = nn.Linear(fc_fan_in_dim, fc_fan_out_dim, device=f"cuda:{rank}")

    def forward(self, x):
        loginfo(self.rank, f"input shape={x.shape}")
        x = self.embedding(x)
        assert x.requires_grad
        loginfo(self.rank, f"embedding shape={x.shape}")
        x = torch.reshape(x, x.shape[:-2] + (-1,))
        assert x.requires_grad
        loginfo(self.rank, f"concated shape={x.shape}")
        for hidden_layer, relu in zip(self.hidden_layers, self.relus):
            x = self.__class__.column_linear_comm(x)  # All gather
            assert x.requires_grad
            # x.shape == [..., H_in]
            x = hidden_layer(x)
            assert x.requires_grad
            # x.shape == [..., H_out / W]
            loginfo(self.rank, f"hidden layer shape={x.shape}")
            x = relu(x)
            assert x.requires_grad
            loginfo(self.rank, f"relu output shape={x.shape}") 
        loginfo(self.rank, f"last layer shape={x.shape}")
        x = self.__class__.column_linear_comm(x)
        assert x.requires_grad
        x = self.fc(x)
        assert x.requires_grad
        loginfo(self.rank, f"output shape={x.shape}")
        x = self.__class__.column_linear_comm(x)
        assert x.requires_grad
        logits = x[..., :self.num_classes]
        assert logits.requires_grad
        return logits
        


def train(
    model,
    input_tensor,
    label_tensor,
    optimizer,
    batch_size=256,
    batch_stride=None,
    batch_offset=0,
    epochs=1,
    steps=None,
    logger=logging.info,
    rank=None,  # for debugging only
):
    if batch_stride is None:
        batch_stride = batch_size
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for i in range(0, input_tensor.shape[0] - batch_stride, batch_stride):
            batch_input_tensor = input_tensor[i + batch_offset : i + batch_offset + batch_size]
            # logging.info(f"batch_input_tensor.shape={batch_input_tensor.shape}")
            batch_label_tensor = label_tensor[i + batch_offset : i + batch_offset + batch_size]
            # logging.info(f"batch_label_tensor.shape={batch_label_tensor.shape}")
            optimizer.zero_grad()
            # logger(f"Starting forward. i={i}")
            # if rank > 1:
            #     time.sleep(2)
            output = model(batch_input_tensor)
            # logger(f"Finished forward. i={i}")
            # logging.info(f"output.shape={output.shape}")
            loss = loss_fn(output, batch_label_tensor).mean()
            # logging.info(f"loss.shape={loss.shape}")
            # logger(f"Starting backward. i={i}")
            # if rank > 1:
            #     time.sleep(2)
            loss.backward()
            # logger(f"Finished backward. i={i}")
            optimizer.step()
            # break
            if i / batch_size % 100 == 0:
                logger(f"Epoch {epoch} step {i / batch_size} loss: {loss.item()}")
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


def create_mlp_e32_512(vocab_size, category_size, model=MLPModel):
    return model(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dims=[512],
        seq_length=max_title_token_length,
        num_classes=category_size,
    )


def ddp_main_fn(
    rank,
    world_size,
    vocab_size,
    category_size,
    train_input_tensor,
    train_label_tensor,
):
    def loginfo(msg):
        sys.stdout.write(f"P{rank}] {msg}\n")

    loginfo(f"Starting ddp_main_fn rank {rank} of world_size {world_size} ...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '18121'
    
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    model = create_mlp_e32_512(vocab_size, category_size).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    ddp_optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    batch_stride = 1024
    batch_size = batch_stride // world_size
    assert batch_stride == world_size * batch_size
    batch_offset = rank * batch_size

    train(
        ddp_model,
        train_input_tensor.to(rank),
        train_label_tensor.to(rank),
        ddp_optimizer,
        batch_stride=batch_stride,
        batch_size=batch_size,
        batch_offset=batch_offset,
        logger=loginfo,
    )

    torch.save(ddp_model.state_dict(), f"./checkpoint/ddp_rank{rank}.checkpoint")
    loginfo("model saved")


def tp_main_fn(
    rank,
    world_size,
    vocab_size,
    category_size,
    train_input_tensor,
    train_label_tensor,
    tp_model_cls,
):
    def loginfo(msg):
        t = datetime.datetime.now().strftime("%H:%M:%S.%f")
        sys.stdout.write(f"Rank{rank} {t}] {msg}\n")

    loginfo(f"Starting tp_main_fn rank {rank} of world_size {world_size} ...")
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '18121'
    
    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3))
    model = tp_model_cls(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dims=[512],
        seq_length=max_title_token_length,
        num_classes=category_size,
        rank=rank,
        world_size=world_size,
    )
    tp_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(
        model,
        train_input_tensor.to(rank),
        train_label_tensor.to(rank),
        tp_optimizer,
        batch_size=1024,
        logger=loginfo,
        rank=rank,
    )

    torch.save(model.state_dict(), f"./checkpoint/tp_rank{rank}.checkpoint")
    loginfo("model saved")



def main(_):
    if False:
        logging.info("----------- Load Data from CSV files -----------")
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
        # save vocab, categories, train_input_tensor, train_label_tensor, test_input_tensor, test_label_tensor
        logging.info("----------- Saving Data to Cache -----------")
        torch.save(vocab, "./checkpoint/vocab.pt")
        torch.save(categories, "./checkpoint/categories.pt")
        torch.save(train_input_tensor, "./checkpoint/train_input_tensor.pt")
        torch.save(train_label_tensor, "./checkpoint/train_label_tensor.pt")
        torch.save(test_input_tensor, "./checkpoint/test_input_tensor.pt")
        torch.save(test_label_tensor, "./checkpoint/test_label_tensor.pt")
    elif True:
        logging.info("----------- Load Data from Cache -----------")
        vocab = torch.load("./checkpoint/vocab.pt")
        categories = torch.load("./checkpoint/categories.pt")
        train_input_tensor = torch.load("./checkpoint/train_input_tensor.pt")
        train_label_tensor = torch.load("./checkpoint/train_label_tensor.pt")
        test_input_tensor = torch.load("./checkpoint/test_input_tensor.pt")
        test_label_tensor = torch.load("./checkpoint/test_label_tensor.pt")
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
        mlp_e32_512 = create_mlp_e32_512(len(vocab), len(categories))
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

    if False:
        logging.info("----------- CPU training With Customized Linear -----------")
        mlp_e32_512 = create_mlp_e32_512(len(vocab), len(categories), model=MLPModelWithCustomizedLinear)
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

    if True:
        logging.info("----------- Single GPU training -----------")
        assert torch.cuda.is_available()
        device_train_labels = train_label_tensor.to('cuda:0')
        device_train_inputs = train_input_tensor.to('cuda:0')
        device_mlp_e32_512 = create_mlp_e32_512(len(vocab), len(categories)).to('cuda:0')
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
        dd_mlp_e32_512 = nn.DataParallel(create_mlp_e32_512(len(vocab), len(categories)))
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

    if False:
        logging.info("----------- Distributed Data Parallel ------------")
        world_size = 4
        assert world_size <= torch.cuda.device_count()
        torch.multiprocessing.spawn(
            ddp_main_fn,
            args=(world_size, len(vocab), len(categories), train_input_tensor, train_label_tensor),
            nprocs=world_size,
            join=True,
        )
        # torch.distributed.destroy_process_group()
        state_dict = torch.load("./checkpoint/ddp_rank2.checkpoint")
        adjusted_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        mlp_e32_512 = create_mlp_e32_512(len(vocab), len(categories))
        mlp_e32_512.load_state_dict(adjusted_state_dict)
        logging.info("Evaluating on train set:")
        eval(mlp_e32_512, train_input_tensor, train_label_tensor)
        logging.info("Evaluating on test set:")
        eval(mlp_e32_512, test_input_tensor, test_label_tensor)

    if False:
        logging.info("----------- Tensor Parallel Row Linear ------------")
        world_size = 4
        assert world_size <= torch.cuda.device_count()
        torch.multiprocessing.spawn(
            tp_main_fn,
            args=(
                world_size,
                len(vocab),
                len(categories),
                train_input_tensor,
                train_label_tensor,
                TPMLPModelWithDirectRowLinear,
            ),
            nprocs=world_size,
            join=True,
        )

    if False:
        logging.info("----------- Tensor Parallel Customized Row Linear ------------")
        world_size = 4
        assert world_size <= torch.cuda.device_count()
        torch.multiprocessing.spawn(
            tp_main_fn,
            args=(
                world_size,
                len(vocab),
                len(categories),
                train_input_tensor,
                train_label_tensor,
                TPMLPModelWithCustomizedRowLinear,
            ),
            nprocs=world_size,
            join=True,
        )

    if True:
        logging.info("----------- Tensor Parallel Column Linear ------------")
        world_size = 4
        assert world_size <= torch.cuda.device_count()
        torch.multiprocessing.spawn(
            tp_main_fn,
            args=(
                world_size,
                len(vocab),
                len(categories),
                train_input_tensor,
                train_label_tensor,
                TPMLPModelWithColumnLinear,
            ),
            nprocs=world_size,
            join=True,
        )
    
    logging.info("----------- Done -----------")


if __name__ == "__main__":
    app.run(main)
    
