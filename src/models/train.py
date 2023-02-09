import torch


def train_net(net, optimizer, loss_function, scheduler, train_dataloader, valid_dataloader, test_dataloader):

    #!---------- Training ----------!#

    net.train(True)
    running_loss = 0.0

    # Iterate over batches
    for index_t, batch in enumerate(train_dataloader):

        inputs, targets = batch

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Zero the gradients for every batch
        optimizer.zero_grad()

        # Perform forward pass
        outputs = net(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        running_loss += loss.item()

    #!---------- Validation ----------!#

    net.train(False)
    running_vloss = 0.0

    for index_v, batch in enumerate(valid_dataloader):

        inputs, targets = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = net(inputs)
        loss = loss_function(outputs, targets)
        running_vloss += loss.item()

    #!---------- Calculate average losses ----------!#
    avg_loss = running_loss / (index_t + 1)
    avg_vloss = running_vloss / (index_v + 1)

    # Reduce LR on plateau
    scheduler.step(avg_vloss)

    #!---------- Test ----------!#

    net.train(False)
    predictions = []
    ground_truth = []

    for index, batch in enumerate(test_dataloader):

        inputs, targets = batch
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = net(inputs)
        predictions.append(outputs)
        ground_truth.append(inputs)

    return avg_loss, avg_vloss, predictions, ground_truth
