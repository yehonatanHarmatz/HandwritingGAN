#TODO- for loop with 16bit that trains a vgg19bn to recognize writers
from torch import optim
from torch.cuda.amp import GradScaler, autocast

from models.StyleEncoder_model import StyleEncoder


def freeze_conv(model):
    for param in model.features:
        param.requires_grad = False

def train_vgg_extractor():

    # Creates model and optimizer in default precision
    #model = StyleEncoder()
    #create a new encoder
    #replace the last layer with a writers number

    #freeze some of the layers

    model = StyleEncoder(False)
    #)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), ...)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    epochs=20
    #TODO- put harmatz dataset here
    data=None
    loss_fn=None
    #argmax()==label
    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                # size of height 32 * K , width , channel=3
                output = model(input['image'])
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