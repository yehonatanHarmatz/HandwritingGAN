#TODO- for loop with 16bit that trains a vgg19bn to recognize writers
from torch import optim
from torch.cuda.amp import GradScaler, autocast


def train_vgg_extractor(model):
    #replace the last layer with a writers number

    # Creates model and optimizer in default precision
    #model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    epochs=20
    #TODO- put harmatz dataset here
    data=None
    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()