import torch
import numpy as np

#initializing a Tensor

#tensors can be created directly from data. data type is automatically inferred
data = [[1,2], [3,4]]
x_data = torch.tensor(data)


#tensors can be created from numpy arraws and vice versa

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) #retains the properties of x_data
print(f"Ones Tensor: \n {x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float) #overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")

#we move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    rand_tensor = rand_tensor.to(
        torch.accelerator.current_accelerator()
    )
    print(rand_tensor)
    

#standard numpy-like indexing and slicing:
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

#meaning of this ":" here is all rows, select everything along this dimension
# and the "1" is column 1
tensor[:,1] = 0
print(f"{tensor}\n")

#Joining tensors we can use torch.cat to concatenate a sequence of tensors along given dimensions
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"this is cat: {t1}")



#arithmetic operations
#This computes the matix multiplication between two tensors. y1, y2,y2 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

#this computes the element-wise product. z1, z2,z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

