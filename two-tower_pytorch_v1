# Imports
from google.colab import drive
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from statistics import mean

# Mount Google Drive to access the dataset
drive.mount("/content/drive")

# Define file paths for the dataset
rating_file_path = "/content/drive/MyDrive/colab/recom_sys/data/ml-20m/ratings.csv"
genre_file_path = "/content/drive/MyDrive/colab/recom_sys/data/ml-20m/movies.csv"

# Load and preprocess the dataset
ratings_df = pd.read_csv(rating_file_path)
movies_df = pd.read_csv(genre_file_path)

# Extract relevant columns
user_movie_ratings = ratings_df[["userId", "movieId", "rating"]].copy()
movie_genres = movies_df[['movieId', 'genres']].copy()

# Convert genres to one-hot encoding
genre_one_hot = movie_genres['genres'].str.get_dummies(sep='|')
movie_genres = movie_genres[['movieId']].join(genre_one_hot).set_index('movieId')
genre_dict = movie_genres.apply(lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1).to_dict()

# Split dataset into training and testing sets
X = user_movie_ratings[["userId", "movieId"]].values
y = user_movie_ratings[["rating"]].values
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)

# Convert to PyTorch tensors
train_X = torch.tensor(train_X, dtype=torch.int64)
test_X = torch.tensor(test_X, dtype=torch.int64)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# Define the MovieLens Dataset class
class MovieLensDataset(Dataset):
    def __init__(self, user_item_matrix, ratings, genre_features):
        self.user_item_matrix = user_item_matrix
        self.ratings = ratings
        self.genre_features = genre_features

    def __len__(self):
        return len(self.user_item_matrix)

    def __getitem__(self, idx):
        user_id, item_id = self.user_item_matrix[idx]
        rating = self.ratings[idx]
        genre_feature = self.genre_features.get(item_id.item(), torch.zeros(len(genre_one_hot.columns), dtype=torch.float32))
        return user_id, item_id, genre_feature, rating

# Create dataset and data loaders
num_genres = len(genre_one_hot.columns)
train_dataset = MovieLensDataset(train_X, train_y, genre_dict)
test_dataset = MovieLensDataset(test_X, test_y, genre_dict)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# Define the User and Item Network classes
class UserNet(nn.Module):
    def __init__(self, num_users, embedding_dim=20):
        super(UserNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_ids):
        return self.user_embedding(user_ids)

class ItemNet(nn.Module):
    def __init__(self, num_items, num_genres, embedding_dim=20):
        super(ItemNet, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_layer = nn.Linear(num_genres, num_genres - 1)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + num_genres - 1, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, item_ids, genre_features):
        item_embeddings = self.item_embedding(item_ids)
        genre_embeddings = self.genre_layer(genre_features)
        combined_embeddings = torch.cat([item_embeddings, genre_embeddings], dim=1)
        return self.mlp(combined_embeddings)

# Instantiate networks and optimizer
max_user_id, max_item_id = X.max(axis=0)
user_net = UserNet(max_user_id + 1)
item_net = ItemNet(max_item_id + 1, num_genres)

optimizer = optim.Adam(list(user_net.parameters()) + list(item_net.parameters()), lr=0.001)
loss_fn = nn.MSELoss()

# Move networks to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
user_net.to(device)
item_net.to(device)

# Evaluation function
def eval_net(user_net, item_net, loader, device="cpu"):
    user_net.eval()
    item_net.eval()
    loss_fn = nn.MSELoss(reduction='sum')  # Use reduction='sum' to accumulate the total loss
    total_loss = 0.0
    total_samples = 0  # Keep track of the total number of samples

    with torch.no_grad():
        for user_ids, item_ids, genre_features, ratings in loader:
            user_ids, item_ids, genre_features, ratings = user_ids.to(device), item_ids.to(device), genre_features.to(device), ratings.to(device)
            user_embeddings = user_net(user_ids)
            item_embeddings = item_net(item_ids, genre_features)
            predictions = (user_embeddings * item_embeddings).sum(dim=1)
            loss = loss_fn(predictions, ratings.view(-1))
            total_loss += loss.item()
            total_samples += ratings.size(0)  # Increment the total number of samples

    return total_loss / total_samples  # Return the average loss per sample


# Training loop
for epoch in range(4):
    user_net.train()
    item_net.train()
    running_loss = []
    for user_ids, item_ids, genre_features, ratings in tqdm(train_loader):
        user_ids, item_ids, genre_features, ratings = user_ids.to(device), item_ids.to(device), genre_features.to(device), ratings.to(device)
        optimizer.zero_grad()
        user_embeddings = user_net(user_ids)
        item_embeddings = item_net(item_ids, genre_features)
        predictions = (user_embeddings * item_embeddings).sum(dim=1)
        loss = loss_fn(predictions, ratings.view(-1))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    train_loss = mean(running_loss)
    test_loss = eval_net(user_net, item_net, test_loader, device=device)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Example prediction
user_net.to("cpu")
item_net.to("cpu")
user_id = torch.tensor([1], dtype=torch.int64)
item_id = torch.tensor([10], dtype=torch.int64)
genre_feature = genre_dict[10].view(1, -1)
user_embedding = user_net(user_id)
item_embedding = item_net(item_id, genre_feature)
prediction = (user_embedding * item_embedding).sum(dim=1)
print(f"Predicted rating for user {user_id.item()} and item {item_id.item()}: {prediction.item():.4f}")
