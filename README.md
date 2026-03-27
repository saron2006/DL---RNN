# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="1327" height="277" alt="image" src="https://github.com/user-attachments/assets/9ea02615-2b12-45e6-9ba2-c9e0d432f971" />

## DESIGN STEPS

### STEP 1: Load and normalize data, create sequences.
### STEP 2: Convert data to tensors and set up DataLoader.
### STEP 3: Define the RNN model architecture.
### STEP 4: Summarize, compile with loss and optimizer.
### STEP 5: Train the model with loss tracking.
### STEP 6: Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: Saron Xavier A
### Register Number: 212223230197

```python
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="696" height="495" alt="image" src="https://github.com/user-attachments/assets/05c1914c-a4a6-47ef-aad3-d481c6b2bd66" />

## True Stock Price, Predicted Stock Price vs time
<img width="890" height="575" alt="image" src="https://github.com/user-attachments/assets/62804cb0-3393-4e9f-9dae-47b332cab592" />


### Predictions
<img width="1362" height="60" alt="image" src="https://github.com/user-attachments/assets/ca4e16b2-16b0-40c7-b22b-12d8dc670e39" />

## RESULT
Thus, the developing a recurrent neural network model for stock prediction was executed successfully.
