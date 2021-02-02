import torch
from torch import nn
from torch.nn import functional as F
from super_selfish.utils import bcolors
from tqdm import tqdm
from colorama import Fore


class BaseTrainer():

    def __init__(self, model, dataset=None, loss=nn.CrossEntropyLoss(reduction='mean'), collate_fn=None):
        """Constitutes a self-supervision algorithm. All implemented algorithms are childs. Handles training, storing, and
        loading of the trained model/backbone.

        Args:
            model (torch.nn.Module): The module to self supervise.
            dataset (torch.utils.data.Dataset): The dataset to train on.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
            collate_fn (function, optional): The collate function. Defaults to None.
        """
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.collate_fn = collate_fn

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=True,
                  num_workers=0, name="store/base", pretrained=False, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)):
        """Starts the training procedure of a self-supervision algorithm.

        Args:
            lr (float, optional): Optimizer learning rate. Defaults to 1e-3.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler. Defaults to lambdaoptimizer:torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0).
            optimizer (torch.optim.Optimizer, optional): Optimizer to use. Defaults to torch.optim.Adam.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Size of bachtes to process. Defaults to 32.
            shuffle (bool, optional): Wether to shuffle the dataset. Defaults to True.
            num_workers (int, optional): Number of workers to use. Defaults to 0.
            name (str, optional): Path to store and load models. Defaults to "store/base".
            pretrained (bool, optional): Wether to load pretrained model. Defaults to False.
        """
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            raise("No dataset has been specified.")

        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        self._load_pretrained(name, pretrained)
        try:
            train_loader, optimizer, lr_scheduler = self._init_data_optimizer(
                optimizer=optimizer, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn, lr=lr, lr_scheduler=lr_scheduler)
            self._epochs(epochs=epochs, train_loader=train_loader,
                         optimizer=optimizer, lr_scheduler=lr_scheduler)
        finally:
            self.save(name)
            print()

    def _load_pretrained(self, name, pretrained):
        """Private method to load a pretrained model

        Args:
            name (str): Path to model.
            pretrained (bool): Wether to load pretrained model.

        Raises:
            IOError: [description]
        """
        try:
            if pretrained:
                self.load(name)
        except Exception:
            raise IOError("Could not load pretrained.")

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        """Creates all objects that are neccessary for the self-supervision training and are dependend on self.supervise(...).

        Args:
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
            batch_size (int, optional): Size of bachtes to process.
            shuffle (bool, optional): Wether to shuffle the dataset.
            num_workers (int, optional): Number of workers to use.
            collate_fn (function, optional): The collate function.
            lr (float, optional): Optimizer learning rate.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.

        Returns:
            Tuple: All created objects
        """
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
        optimizer = optimizer(self.model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler(optimizer)

        return train_loader, optimizer, lr_scheduler

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        """Implements the training loop (epochs, batches, visualization) excluding the actual training step.

        Args:
            epochs (int, optional): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): Iterator over the dataset.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                Fore.GREEN, Fore.RESET), desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size:
                    continue
                optimizer.zero_grad()

                loss = self._forward(data)
                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer,
                             lr_scheduler=lr_scheduler)

    def _forward(self, data):
        """Forward part of training step. Conducts all forward calculations.

        Args:
            data (Tuple(torch.FloatTensor,torch.FloatTensor)): Batch of instances with corresponding labels.

        Returns:
            torch.FloatTensor: Loss of batch.
        """
        inputs, labels = data
        outputs = self.model(inputs.to('cuda'))[0]
        loss = self.loss(outputs, labels.to('cuda'))
        return loss

    def _update(self, loss, optimizer, lr_scheduler):
        """Backward part of training step. Calculates gradients and conducts optimization step.
        Also handles other updates like lr scheduler.

        Args:
            loss (torch.nn.Module, optional): The critierion to train on.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    def to(self, name):
        """Wraps device handling.

        Args:
            name (str): Name of device, see pytorch.

        Returns:
            Supervisor: Returns itself.
        """
        self.model = self.model.to(name)
        return self

    def save(self, name="store/base"):
        """Saves model parameters to disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        torch.save(self.model.state_dict(), name + ".pt")
        print(bcolors.OKBLUE + "Saved at " + name + "." + bcolors.ENDC)
        return self

    def load(self, name="store/base"):
        """Loads model parameters from disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        pretrained_dict = torch.load(name + ".pt")
        print(bcolors.OKBLUE + "Loaded", name + "." + bcolors.ENDC)
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        return self
