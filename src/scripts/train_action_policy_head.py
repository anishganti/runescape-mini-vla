import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#current setup is very simple 1 layer MLP + 4 heads with gating logic + cross entropy loss
#need to read + tweak MLP #layers #heads + loss function + optimizers
#then need to change architecture to maybe transformer/diffusion down the road

base_dir = "/Users/anishganti/runescape_mini_vla/src/data/mining"

class ActionPolicyHead(nn.Module):
    def __init__(self):
        super(ActionPolicyHead, self).__init__()
        self.hidden = nn.Linear(576,256)
        self.relu = nn.ReLU()

        #self.output = nn.Linear(256, 4) -- this will not work because our labels need to output different sets of classes. 
        self.head_a = nn.Linear(256,3)
        self.head_k = nn.Linear(256,3)
        self.head_x = nn.Linear(256,20)
        self.head_y = nn.Linear(256,20)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        
        output_a = self.head_a(x)
        output_k = self.head_k(x)
        output_x = self.head_x(x)
        output_y = self.head_y(x)

        return output_a, output_k, output_x, output_y

def get_episodes(path):
    episodes = [
        episode for episode in os.listdir(path)
        if os.path.isdir(os.path.join(path, episode))
    ]
    
    return episodes

def load_embeddings(episode):
    file_name = f"/Users/anishganti/runescape_mini_vla/src/data/mining/{episode}/{episode}_embeddings.pt"
    embeddings = torch.load(file_name)
    embeddings = torch.stack(embeddings)
    return embeddings

def load_actions(episode):
    file_name = f"/Users/anishganti/runescape_mini_vla/src/data/mining/{episode}/{episode}.json"

    with open(file_name, "r", encoding="utf-8") as file_handle:
        events = json.load(file_handle)
        actions = events['events']        
        return actions

def encode_actions(actions):
    a_tensor = []
    k_tensor = []
    x_tensor = []
    y_tensor = []

    for action in actions: 
        a_tensor.append(action['a'])
        k_tensor.append(action.get('k', -1))
        x_tensor.append(action.get('x', -1))
        y_tensor.append(action.get('y', -1))

    a_tensor = torch.tensor(a_tensor)
    k_tensor = torch.tensor(k_tensor)
    x_tensor = torch.tensor(x_tensor)
    y_tensor = torch.tensor(y_tensor)

    return a_tensor, k_tensor, x_tensor, y_tensor
    
def pool_embeddings(embeddings): 
    return embeddings.mean(dim=1)

def init_model():
    model = ActionPolicyHead()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

def save_model(model):
    torch.save(model.state_dict(), "/Users/anishganti/runescape_mini_vla/src/models/mlp/checkpoint.pt")

def train(loader, model, criterion, optimizer):
    print("Training model...")
    num_epochs = 30
    total_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for embedding, action_a, action_k, action_x, action_y in loader:
            a,k,x,y = model(embedding)

            loss_a = criterion(a, action_a)
            loss_k = criterion(k, action_k)
            loss_x = criterion(x, action_x)
            loss_y = criterion(y, action_y)

            loss = loss_a + loss_k + loss_x + loss_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def main():
    episodes = get_episodes(base_dir)

    action_a_tensors = []
    action_k_tensors = []
    action_x_tensors = []
    action_y_tensors = []
    embedding_tensors = []

    for episode in episodes:

        actions = load_actions(episode)
        a_tensor, k_tensor, x_tensor, y_tensor = encode_actions(actions)
        action_a_tensors.append(a_tensor[1:])
        action_k_tensors.append(k_tensor[1:])
        action_x_tensors.append(x_tensor[1:])
        action_y_tensors.append(y_tensor[1:])

        embeddings = load_embeddings(episode)
        embeddings = pool_embeddings(embeddings)
        embedding_tensors.append(embeddings[:-1])

    action_a_tensors = torch.cat(action_a_tensors, dim=0)
    action_k_tensors = torch.cat(action_k_tensors, dim=0)
    action_x_tensors = torch.cat(action_x_tensors, dim=0)
    action_y_tensors = torch.cat(action_y_tensors, dim=0)
    embedding_tensors = torch.cat(embedding_tensors, dim=0)

    dataset = TensorDataset(
        embedding_tensors,
        action_a_tensors,
        action_k_tensors,
        action_x_tensors,
        action_y_tensors
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model, criterion, optimizer = init_model()
    model = train(loader, model, criterion, optimizer)
    save_model(model)
