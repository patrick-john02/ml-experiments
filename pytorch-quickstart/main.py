import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 #Torch Vision transform API



#download training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

#download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) #v2.ToImage() converts the loaded image into a PyTorch image tensor so subsequent transforms can work with it.
    #Compose means Take several transformations and execute them one after another.
    #results:
    # Tensor
    
    # shape = [1, 28, 28] - FashionMNIST images are grayscale:
    #Channels x Height x Width 
    
    #For an RGB image:
    # 3 x 224 x 224 - RGB
)




batch_size = 64

#create a loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N,C,H,W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#CREATING MODELS
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"User {device}")

#lets define a model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(), #turns all negatives numbers into flat zero
            
            nn.Linear(512, 512), #512 here is neurons, hidden layer
            #question: Why is the next one nn.Linear(512, 512)?
            #Because the previous layer outputs 512 values.
            #we couldnt write other numbers because the previous process produces 512 values.
            
            nn.ReLU(), #turns all negatives numbers into flat zero
            nn.Linear(512, 10) #FashionMNIST has 10 classes.
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)



# Measures how wrong the model's classification predictions are
loss_function = nn.CrossEntropyLoss()

# Updates the model's parameters using Stochastic Gradient Descent
# lr=1e-3 means learning rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# Function responsible for training the model for one pass through the DataLoader
def train(dataloader, model, loss_function, optimizer):

    # Total number of samples/images in the dataset
    size = len(dataloader.dataset) #so on FashionMNIST has 60,000 images. this means size = 60000

    # Put the neural network into training mode
    model.train() #this does not actually perform the training, it switches certain layers into their training behavior.
    #this becomes especially important with layers such as: Dropout, BatchNorm.


    # Loop through the DataLoader one batch at a time
    for batch, (X,y) in enumerate(dataloader):
        # Move both the input data and labels to the selected device
        X, y = X.to(device), y.to(device)
        
        
        #compute prediction errror
        # Send the current batch through the neural network
        prediction = model(X) #this runs the forward() randomized floats e.g. [0.2, 0.6, 1.6, 4.7...]
        #X → model(X) → logits

        # Compare the predictions against the correct labels
        loss = loss_function(prediction, y) 
        #logits + correct labels → loss

        
        #backpropagation
        loss.backward() # Calculate gradients: determine how each parameter contributed to the loss
        optimizer.step() # Update the model parameters using those gradients
        optimizer.zero_grad() # Clear the gradients before processing the next batch
        
        if batch % 100 == 0: # batch 100 -> print
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


#we also check the model's performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_function(prediction,y).item()
            correct +=(prediction.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Errror:  \n Accuracy: {(100*correct):>0.1f}%, Average Loss: {test_loss:>8f} \n")




#The training process is conducted over several iteration (epochs). During each epoch, the model learns parameters to make better predictions.
#We print the model's accuracy and loss at each epoch; we'd like to see theaccuracy increase and the loss decrease with every epoch. 

epoch = 2
for t in range(epoch):
    print(f"Epoch {t+1}\n -------")
    train(train_dataloader, model, loss_function, optimizer)
    test(test_dataloader, model, loss_function)

print("Done!")



#saving Models
torch.save(model.state_dict(), "modelk.pth")
print("Save Pytorch Model State to model.pth")


#so this model can be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

