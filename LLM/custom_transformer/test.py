import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Transformer(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dim: int,
        output_dim: int,
        num_head: int,
        num_layers: int,
    ) -> None:
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(input_dims, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(hidden_dim, num_head),
            num_layers=num_layers,
        )
        self.memory = torch.rand(10, 32, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        # >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # >>> memory = torch.rand(10, 32, 512)
        # >>> tgt = torch.rand(20, 32, 512)
        # >>> out = transformer_decoder(tgt, memory)
        # [64, 1, 28, 28]
        # print(f"input shape: {x.shape}")
        x = torch.flatten(x, start_dim=1)
        # [64, 784]
        # print(f"flatten: {x.shape}")
        x = self.embedding(x)
        # [64, 784, 256]
        # print(x.shape)
        # 320x512 and 256x512
        x = self.decoder(x, self.memory)
        x = torch.mean(x, dim=1)  # mean pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device).long()
        target = target.to(device).long()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.long()
            target = target.long()

            output = model(data)
            test_loss += torch.max(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)
    return test_loss, test_acc


if __name__ == "__main__":
    # 定义模型参数
    input_dim = 1000  # 输入维度（词汇表大小）
    hidden_dim = 256  # 隐藏单元维度
    output_dim = 10  # 输出维度（类别数）
    num_heads = 8  # 多头注意力的头数
    num_layers = 4  # Transformer 层数
    num_epochs = 10

    # 定义数据变换
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ]
    )

    # 加载MNIST训练集和测试集
    train_dataset = datasets.MNIST(
        root="./dataset", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./dataset", train=False, transform=transform, download=True
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    # else:
    device = torch.device("cpu")

    # 实例化模型
    model = Transformer(input_dim, hidden_dim, output_dim, num_heads, num_layers)
    model

    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 进行训练和评估
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, criterion)
        loss, acc = evaluate(model, device, test_loader)
        print(f"Epoch: {epoch+1}, Accuracy: {acc}% Loss: {loss}")
