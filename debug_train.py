import torch
import sys, os, time, datetime
from torch import nn
from multiprocessing import Pool


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        world_size = torch.distributed.get_world_size()
        all_input = [torch.zeros_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(all_input, input)
        return torch.cat(all_input, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # Prepare for the reduce scatter [B, H] -> [WorldSize, B, H/WorldSize]
        # this assumption is not necessary if we use torch.permute instead of torch.transpose
        # assert len(grad_output.shape) == 2
        # x = torch.reshape(grad_output, grad_output.shape[:-1] + (world_size, grad_output.shape[-1] // world_size))
        # x = torch.transpose(x, 0, -2)
        
        # y = torch.zeros_like(x[0])
        # torch.distributed.reduce_scatter_tensor(y, x, op=torch.distributed.ReduceOp.SUM)
        # return y
        stride = grad_output.shape[-1] // world_size
        return grad_output[..., rank * stride:(rank + 1) * stride]
        

class ColumnLinear(nn.Module):
    
    def __init__(self,
                 feature_dim,
                 num_classes,
                 rank,
                 world_size,
                ):
        super().__init__()
        self.world_size = world_size
        assert num_classes % world_size == 0
        self.fc = nn.Linear(feature_dim, num_classes // world_size, bias=(rank == 0), device=f"cuda:{rank}")
        # overwrite init so that the TP and single device model have the same init
        with torch.no_grad():
            self.fc.weight = torch.nn.Parameter(torch.randn_like(self.fc.weight))
        
    def forward(self, x):
        x = self.fc(x)
        y = AllGather.apply(x)
        return y


def train(
    model,
    input_tensor,
    label_tensor,
    optimizer,
    epochs=30,
    batch_size=64,
    rank=0,
):
    assert input_tensor.shape[0] == label_tensor.shape[0]
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for i in range(0, input_tensor.shape[0] - batch_size, batch_size):
            batch_input_tensor = input_tensor[i : i + batch_size]
            batch_label_tensor = label_tensor[i : i + batch_size]
            # print(f"DEBUG batch_label_tensor={batch_label_tensor}")

            optimizer.zero_grad()
            output = model(batch_input_tensor)
            # print(f"DEBUG output.shape={output.shape}")
            loss = loss_fn(output, batch_label_tensor).mean()
            loss.backward()
            optimizer.step()
            if rank == 0 and i / batch_size % 10 == 0:
                print(f"Epoch {epoch} step {i / batch_size} loss: {loss.item()}")
        end_time = time.time()
    if rank == 0:
        print(f"Final loss: {loss.item()} time: {(end_time - start_time):.2f}s")
    return


def tp_main_fn(
    rank,
    world_size,
    num_categories,
    input_tensor,
    label_tensor,
):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['NCCL_BLOCKING_WAIT'] = '1'
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '18121'
    
    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3))
    model = ColumnLinear(
        input_tensor.shape[1],
        num_categories,
        rank,
        world_size,
    )
    tp_optimizer = torch.optim.Adam(model.parameters(), lr=0.2)

    assert (label_tensor >= 0).float().mean().item() == 1
    assert (label_tensor < num_categories).float().mean().item() == 1
    # print(f"label_tensor={label_tensor}")

    train(
        model,
        input_tensor.to(device=f"cuda:{rank}"),
        label_tensor.to(device=f"cuda:{rank}"),
        tp_optimizer,
        rank=rank,
    )


def main(_):
    print("----------- Generate Random Data for Debugging -----------")
    num_categories = 16
    example_num = 1000
    feature_dimension = 2
    label_tensor = torch.randint(
        low=0, high=num_categories, size=(example_num, ), dtype=torch.long)
    # Debugging: intentially leak the labels into the input features.
    noise_input_tensor = torch.rand(size=(example_num, feature_dimension - 1))
    input_tensor = torch.cat([noise_input_tensor, label_tensor.unsqueeze(1) * 0.1], dim=-1)

    print("----------- Single GPU training -----------")
    assert torch.cuda.is_available()
    model = nn.Linear(feature_dimension, num_categories, device='cuda:0')
    # overwrite init so that the TP and single device model have the same init
    with torch.no_grad():
        model.weight = torch.nn.Parameter(torch.randn_like(model.weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    train(
        model,
        input_tensor.to('cuda:0'),
        label_tensor.to('cuda:0'),
        optimizer,
    )

    print("----------- Tensor Parallel Row Linear ------------")
    world_size = 8
    assert world_size <= torch.cuda.device_count()
    torch.multiprocessing.spawn(
        tp_main_fn,
        args=(
            world_size,
            num_categories,
            input_tensor,
            label_tensor,
        ),
        nprocs=world_size,
        join=True,
    )

    print("----------- Done -----------")


if __name__ == "__main__":
    main(None)

# OUTPUT:

# ----------- Generate Random Data for Debugging -----------
# ----------- Single GPU training -----------
# Epoch 0 step 0.0 loss: 3.0204732418060303
# Epoch 0 step 10.0 loss: 2.519120216369629
# Epoch 1 step 0.0 loss: 2.39621901512146
# Epoch 1 step 10.0 loss: 2.273987293243408
# Epoch 2 step 0.0 loss: 2.195404529571533
# Epoch 2 step 10.0 loss: 2.107940196990967
# Epoch 3 step 0.0 loss: 2.0847623348236084
# Epoch 3 step 10.0 loss: 2.025460958480835
# Epoch 4 step 0.0 loss: 2.0213863849639893
# Epoch 4 step 10.0 loss: 1.9600309133529663
# Epoch 5 step 0.0 loss: 1.975163221359253
# Epoch 5 step 10.0 loss: 1.9054245948791504
# Epoch 6 step 0.0 loss: 1.9291635751724243
# Epoch 6 step 10.0 loss: 1.8611923456192017
# Epoch 7 step 0.0 loss: 1.894805908203125
# Epoch 7 step 10.0 loss: 1.8244102001190186
# Epoch 8 step 0.0 loss: 1.8637259006500244
# Epoch 8 step 10.0 loss: 1.791170597076416
# Epoch 9 step 0.0 loss: 1.8348579406738281
# Epoch 9 step 10.0 loss: 1.7608898878097534
# Epoch 10 step 0.0 loss: 1.8090931177139282
# Epoch 10 step 10.0 loss: 1.7335267066955566
# Epoch 11 step 0.0 loss: 1.7856695652008057
# Epoch 11 step 10.0 loss: 1.7084163427352905
# Epoch 12 step 0.0 loss: 1.7639718055725098
# Epoch 12 step 10.0 loss: 1.6851552724838257
# Epoch 13 step 0.0 loss: 1.74386465549469
# Epoch 13 step 10.0 loss: 1.6635416746139526
# Epoch 14 step 0.0 loss: 1.7251194715499878
# Epoch 14 step 10.0 loss: 1.6433463096618652
# Epoch 15 step 0.0 loss: 1.707527756690979
# Epoch 15 step 10.0 loss: 1.624388337135315
# Epoch 16 step 0.0 loss: 1.6909632682800293
# Epoch 16 step 10.0 loss: 1.6065220832824707
# Epoch 17 step 0.0 loss: 1.6753082275390625
# Epoch 17 step 10.0 loss: 1.5896261930465698
# Epoch 18 step 0.0 loss: 1.6604632139205933
# Epoch 18 step 10.0 loss: 1.5735994577407837
# Epoch 19 step 0.0 loss: 1.6463470458984375
# Epoch 19 step 10.0 loss: 1.5583550930023193
# Epoch 20 step 0.0 loss: 1.632888674736023
# Epoch 20 step 10.0 loss: 1.5438185930252075
# Epoch 21 step 0.0 loss: 1.6200282573699951
# Epoch 21 step 10.0 loss: 1.5299263000488281
# Epoch 22 step 0.0 loss: 1.6077136993408203
# Epoch 22 step 10.0 loss: 1.5166220664978027
# Epoch 23 step 0.0 loss: 1.5958988666534424
# Epoch 23 step 10.0 loss: 1.5038574934005737
# Epoch 24 step 0.0 loss: 1.5845446586608887
# Epoch 24 step 10.0 loss: 1.4915896654129028
# Epoch 25 step 0.0 loss: 1.5736149549484253
# Epoch 25 step 10.0 loss: 1.4797805547714233
# Epoch 26 step 0.0 loss: 1.5630791187286377
# Epoch 26 step 10.0 loss: 1.4683966636657715
# Epoch 27 step 0.0 loss: 1.552909016609192
# Epoch 27 step 10.0 loss: 1.4574073553085327
# Epoch 28 step 0.0 loss: 1.5430798530578613
# Epoch 28 step 10.0 loss: 1.4467864036560059
# Epoch 29 step 0.0 loss: 1.5335693359375
# Epoch 29 step 10.0 loss: 1.4365088939666748
# Final loss: 1.5271655321121216 time: 0.53s
# ----------- Tensor Parallel Row Linear ------------
# Epoch 0 step 0.0 loss: 3.505143880844116
# Epoch 0 step 10.0 loss: 2.7928383350372314
# Epoch 1 step 0.0 loss: 2.6236414909362793
# Epoch 1 step 10.0 loss: 2.473978042602539
# Epoch 2 step 0.0 loss: 2.357084274291992
# Epoch 2 step 10.0 loss: 2.3594472408294678
# Epoch 3 step 0.0 loss: 2.265211820602417
# Epoch 3 step 10.0 loss: 2.321305274963379
# Epoch 4 step 0.0 loss: 2.2313804626464844
# Epoch 4 step 10.0 loss: 2.2950940132141113
# Epoch 5 step 0.0 loss: 2.208418846130371
# Epoch 5 step 10.0 loss: 2.2768657207489014
# Epoch 6 step 0.0 loss: 2.1957530975341797
# Epoch 6 step 10.0 loss: 2.265079975128174
# Epoch 7 step 0.0 loss: 2.1835649013519287
# Epoch 7 step 10.0 loss: 2.2532334327697754
# Epoch 8 step 0.0 loss: 2.174255132675171
# Epoch 8 step 10.0 loss: 2.2441000938415527
# Epoch 9 step 0.0 loss: 2.16552996635437
# Epoch 9 step 10.0 loss: 2.2358548641204834
# Epoch 10 step 0.0 loss: 2.158254623413086
# Epoch 10 step 10.0 loss: 2.228457450866699
# Epoch 11 step 0.0 loss: 2.1516683101654053
# Epoch 11 step 10.0 loss: 2.2218422889709473
# Epoch 12 step 0.0 loss: 2.1458122730255127
# Epoch 12 step 10.0 loss: 2.2158076763153076
# Epoch 13 step 0.0 loss: 2.1404900550842285
# Epoch 13 step 10.0 loss: 2.2102601528167725
# Epoch 14 step 0.0 loss: 2.1355957984924316
# Epoch 14 step 10.0 loss: 2.2051374912261963
# Epoch 15 step 0.0 loss: 2.1310818195343018
# Epoch 15 step 10.0 loss: 2.200373888015747
# Epoch 16 step 0.0 loss: 2.126878023147583
# Epoch 16 step 10.0 loss: 2.1959197521209717
# Epoch 17 step 0.0 loss: 2.122952461242676
# Epoch 17 step 10.0 loss: 2.1917378902435303
# Epoch 18 step 0.0 loss: 2.119266986846924
# Epoch 18 step 10.0 loss: 2.1877944469451904
# Epoch 19 step 0.0 loss: 2.115795612335205
# Epoch 19 step 10.0 loss: 2.184062957763672
# Epoch 20 step 0.0 loss: 2.1125147342681885
# Epoch 20 step 10.0 loss: 2.180521249771118
# Epoch 21 step 0.0 loss: 2.109405755996704
# Epoch 21 step 10.0 loss: 2.177150011062622
# Epoch 22 step 0.0 loss: 2.1064512729644775
# Epoch 22 step 10.0 loss: 2.1739323139190674
# Epoch 23 step 0.0 loss: 2.103638172149658
# Epoch 23 step 10.0 loss: 2.170854091644287
# Epoch 24 step 0.0 loss: 2.1009538173675537
# Epoch 24 step 10.0 loss: 2.167903423309326
# Epoch 25 step 0.0 loss: 2.0983874797821045
# Epoch 25 step 10.0 loss: 2.165069341659546
# Epoch 26 step 0.0 loss: 2.0959298610687256
# Epoch 26 step 10.0 loss: 2.1623432636260986
# Epoch 27 step 0.0 loss: 2.093574047088623
# Epoch 27 step 10.0 loss: 2.1597163677215576
# Epoch 28 step 0.0 loss: 2.091310977935791
# Epoch 28 step 10.0 loss: 2.157181739807129
# Epoch 29 step 0.0 loss: 2.089134693145752
# Epoch 29 step 10.0 loss: 2.1547329425811768
# Final loss: 2.2350785732269287 time: 9.37s
# ----------- Done -----------
    
