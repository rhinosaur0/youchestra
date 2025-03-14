import torch as th

tensor = th.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])



first_tensor = th.flatten(tensor, start_dim=0, end_dim=1)
second_tensor = th.flatten(tensor.transpose(0, 1), start_dim=0, end_dim=1)

# print(f'first_tensor: {first_tensor}')
# print(f'second_tensor: {second_tensor}')


third_tensor = tensor.reshape(4, -1, 3)
fourth_tensor = tensor.reshape(2, -1, 3)
print(third_tensor)
print(fourth_tensor)
