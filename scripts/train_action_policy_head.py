import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#current setup is very simple 1 layer MLP + 4 heads with gating logic + cross entropy loss
#need to read + tweak MLP #layers #heads + loss function + optimizers
#then need to change architecture to maybe transformer/diffusion down the road

base_dir = "/Users/anishganti/runescape-mini-vla/data/mining"

class ActionPolicyHead(nn.Module):
    def __init__(self):
        super(ActionPolicyHead, self).__init__()
        self.hidden = nn.Linear(576,256)
        self.relu = nn.ReLU()

        #self.output = nn.Linear(256, 4) -- this will not work because our labels need to output different sets of classes. 
        self.head1 = nn.Linear(256,3)
        self.head2 = nn.Linear(256,3)
        self.head3 = nn.Linear(256,20)
        self.head4 = nn.Linear(256,20)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        
        output1 = self.softmax(self.head1(x), dim=1)
        output2 = self.softmax(self.head2(x), dim=1)
        output3 = self.softmax(self.head3(x), dim=1)
        output4 = self.softmax(self.head4(x), dim=1)

        return output1, output2, output3, output4


def get_episodes(path):
    episodes = [
        episode for episode in os.listdir(path)
        if os.path.isdir(os.path.join(path, episode))
    ]
    
    return episodes

def load_embeddings(episode):
    file_name = f"/Users/anishganti/runescape-mini-vla/data/mining/{episode}/{episode}_embeddings.pt"
    embeddings = torch.load(file_name)
    embeddings = torch.stack(embeddings)
    return embeddings

def load_actions(episode):
    file_name = f"/Users/anishganti/runescape-mini-vla/data/mining/{episode}/{episode}.json"

    with open(file_name, "r", encoding="utf-8") as file_handle:
        events = json.load(file_handle)
        actions = events['events']        
        return actions

def encode_actions(actions):
    encoded_actions = []
    for action in actions: 
        a_vec = []
        k_vec = []
        x_vec = []
        y_vec = []

        if "a" in action: 
            action_vec[0] = action['a']
        if "k" in action: 
            action_vec[1] = action['k']
        if "x" in action: 
            action_vec[2] = action['x']
        if "y" in action: 
            action_vec[3] = action['y']

        encoded_actions.append(action_vec)
    
    return torch.tensor(encoded_actions)
    
def pool_embeddings(embeddings): 
    return embeddings.mean(dim=1)

def init_model():
    model = ActionPolicyHead()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

def train(loader, model, criterion, loss):
    num_epochs = 30

    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    episodes = get_episodes(base_dir)

    action_tensors = []
    embedding_tensors = []

    for episode in episodes:

        actions = load_actions(episode)
        actions = encode_actions(actions)
        action_tensors.append(actions[1:])

        embeddings = load_embeddings(episode)
        embeddings = pool_embeddings(embeddings)

        embedding_tensors.append(embeddings[:-1])

    action_tensors = torch.cat(action_tensors, dim=0)
    embedding_tensors = torch.cat(embedding_tensors, dim=0)

    dataset = TensorDataset(embedding_tensors, action_tensors)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model, criterion, optimizer = init_model()

    train(loader, model, criterion, optimizer)

main()