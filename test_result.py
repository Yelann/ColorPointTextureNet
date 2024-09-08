from torch.utils.tensorboard import SummaryWriter 


# out_number_list = [16, 64, 256, 1024]
out_number_list = [1, 2, 3, 4]

# Random
writer = SummaryWriter('./runs/TEST_Random')
loss_list = [4.3, 4.1, 3.75, 0]
for i in range(4):
    writer.add_scalar("Color Loss", loss_list[i], out_number_list[i])
writer.close()

# FPS
writer = SummaryWriter('./runs/TEST_FPS')
loss_list = [4.25, 4.0, 3.5, 2.7]
for i in range(4):
    writer.add_scalar("Color Loss", loss_list[i], out_number_list[i])
writer.close()

# Poisson
writer = SummaryWriter('./runs/TEST_SampleNet')
loss_list = [4.28, 4.05, 3.75, 0]
for i in range(4):
    writer.add_scalar("Color Loss", loss_list[i], out_number_list[i])
writer.close()

# MongeNet
writer = SummaryWriter('./runs/TEST_Uniform')
loss_list = [3.2, 3.2, 3.2, 3.2]
for i in range(4):
    writer.add_scalar("Color Loss", loss_list[i], out_number_list[i])
writer.close()
