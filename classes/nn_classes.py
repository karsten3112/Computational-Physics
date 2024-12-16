import torch

class NNmodel(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_hidden_layers, 
                 dim_hiddens, 
                 activation_func, 
                 bias=True) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.acc_func = activation_func
        self.hidden_layers = torch.nn.ParameterList()
        self.input_layer = torch.nn.Linear(input_dim, dim_hiddens, bias=bias, dtype=torch.float)
        self.output_layer = torch.nn.Linear(dim_hiddens, output_dim, bias=bias, dtype=torch.float)
        for num in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(dim_hiddens, dim_hiddens, bias=bias, dtype=torch.float))

    def forward(self, x):
        x = self.acc_func(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.acc_func(layer(x))
        return self.output_layer(x)


class EarlyStopper():
    def __init__(self, patience=10) -> None:
        self.best_loss = None
        self.patience = patience
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_loss = val_loss


def evaluate_total_loss(model, data_loader, loss_func):
    loss_val = torch.tensor([0.0])
    for x_dat, y_dat in data_loader:
        loss = loss_func(model.forward(x_dat), y_dat)
        loss_val+=loss*x_dat.size(0)
    return loss_val/len(data_loader.dataset)
        


def train_model(epochs, model, train_loader, val_loader, optimizer, test_loader=None, early_stopping=True, patience=10, loss_func=torch.nn.MSELoss()):
    early_stopper = EarlyStopper(patience=patience)
    train_losses = torch.zeros(epochs, 1)
    val_losses = torch.zeros(epochs, 1)
    for epoch in range(epochs):
        model.train()
        for x_dat, y_dat in train_loader: 
            optimizer.zero_grad()
            loss = loss_func(model.forward(x_dat), y_dat) #Remember that the loss calculates a mean loss over all points in batch here.
            train_losses[epoch] += loss.item()*x_dat.size(0) 
            loss.backward()
            optimizer.step()
        
        train_losses[epoch]/=len(train_loader.dataset)
        
        model.eval()
        
        val_losses[epoch]+=evaluate_total_loss(model=model, data_loader=val_loader, loss_func=loss_func)
        
        if early_stopping == True:
            early_stopper(val_losses[epoch])
            if early_stopper.early_stop:
                break
    
    if test_loader is None:
        return train_losses[:epoch+1], val_losses[:epoch+1]
    else:
        test_loss = evaluate_total_loss(model=model, data_loader=test_loader, loss_func=loss_func)
        return train_losses[:epoch+1], val_losses[:epoch+1], test_loss
    
