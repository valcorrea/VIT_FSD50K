import torch
def save_model(epoch: int, val_acc: float, save_path: str, net, optimizer = None, log_file : str = None) -> None:
    """Saves checkpoint.

    Args:
        epoch (int): Current epoch.
        val_acc (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer
    }

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)
    
    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")