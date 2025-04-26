import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def SPRT(S, delta_mu, alpha=0.05, beta=0.1):
    A = torch.tensor(np.log((1 - beta) / alpha), dtype=torch.float32).to(device)
    B = torch.tensor(np.log(beta / (1 - alpha)), dtype=torch.float32).to(device)
    batch_size = S.shape[0]
    predictions = torch.zeros(batch_size, dtype=torch.float32, requires_grad=True).to(device)
    samples_taken = torch.zeros(batch_size, dtype=torch.float32).to(device)
    for i in range(batch_size):
        sample = S[i]
        D1 = int(0.01 * sample.shape[0]) 
        LR = torch.tensor(0.0, dtype=torch.float32).to(device)
        hint = 1  

        j = 1 
        n1 = sample[:D1]
        while True:  
            mu0_pred = torch.tensor(0.1, dtype=torch.float32).to(device) + delta_mu[0]
            mu1_pred = torch.tensor(0.1, dtype=torch.float32).to(device) - delta_mu[1]
            LR_numerator = torch.sum(n1 * torch.log(mu0_pred) + (1 - n1) * torch.log(1 - mu0_pred))
            LR_denominator = torch.sum(n1 * torch.log(mu1_pred) + (1 - n1) * torch.log(1 - mu1_pred))
            LR = LR_numerator / (LR_denominator + 1e-19)

            if LR <= A:
                hint = 0
                break
            elif LR >= B:
                hint = 1
                break
            else:
                j += 1
                if 2 * D1 - int(torch.sum(n1).item()) >= len(sample):
                    break  
                n1 = sample[D1: 2 * D1 - int(torch.sum(n1).item())]
        predictions[i] = hint
        samples_taken[i] = j * D1 

    return predictions, samples_taken

def train_model(model, optimizer, criterion, epoch, epochs, delta_mu):
    model.train()
    running_loss = 0.0
    for i in tqdm(range(10000), ncols=100, desc="Training"):
        N = np.random.randint(1000, 100000)
        gt = np.random.uniform(gt_range[0], gt_range[1])
        sample = np.random.choice([0, 1], size=N, p=[gt, 1 - gt])
        if gt > 0.1:
            gt = 0
        else:
            gt = 1
        inputs = torch.tensor([N], dtype=torch.float32).to(device)
        targets = torch.tensor([gt], dtype=torch.float32).to(device)
        outputs = model(inputs) + torch.tensor(delta_mu, dtype=torch.float32).to(device)
        pred, samples = SPRT(torch.tensor([sample], dtype=torch.float32).to(device), outputs)
        loss = criterion(pred.float(), targets.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    return running_loss / 10000


def evaluate_model(model, criterion, delta_mu):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0
    with torch.no_grad():
        for i in tqdm(range(1000), ncols=100, desc="Evaluating"):
            N = np.random.randint(1000, 100000)
            gt = np.random.uniform(gt_range[0], gt_range[1])
            sample = np.random.choice([0, 1], size=N, p=[gt, 1 - gt])
            if gt > 0.1:
                gt = 0
            else:
                gt = 1
            inputs = torch.tensor([N], dtype=torch.float32).to(device)
            targets = torch.tensor([gt], dtype=torch.float32).to(device)
            outputs = model(inputs) + torch.tensor(delta_mu, dtype=torch.float32).to(device)
            pred, samples = SPRT(torch.tensor([sample], dtype=torch.float32).to(device), outputs)
            loss = criterion(pred.float(), targets.float())
            running_loss += loss.item()
            correct += (pred == targets).sum().item()
            total += 1
            total_samples += samples.item()
    accuracy = correct / total
    average_samples = total_samples / total
    return running_loss / 1000, accuracy, average_samples


if __name__ == "__main__":
    gt_range = (0.05, 0.15)
    input_size = 1
    hidden_size = [16, 64]
    output_size = 2
    learning_rate = 0.001
    epochs = 10
    batch_size = 32
    criterion = nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 灵敏度分析：遍历 delta_mu
    delta_mu_values = np.arange(0.01, 0.11, 0.01)  
    results = []
    for delta_mu in delta_mu_values:
        print(f"Evaluating delta_mu = {delta_mu:.2f}")
        

        model = MLP(input_size, hidden_size, output_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            train_loss = train_model(model, optimizer, criterion, epoch, epochs, [delta_mu, delta_mu])
            test_loss, accuracy, average_samples = evaluate_model(model, criterion, [delta_mu, delta_mu])
            print(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}, Average Samples: {average_samples:.2f}"
            )
        
        results.append((delta_mu, accuracy, average_samples))

    print("\nSensitivity Analysis Results:")
    print("Delta Mu | Accuracy | Average Samples")
    print("----------|----------|----------------")
    for delta_mu, accuracy, average_samples in results:
        print(f"{delta_mu:.2f}       | {accuracy:.4f}   | {average_samples:.2f}")

