from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
import numpy as np


def get_lr(model,optimizer,criterion,train_loader,end_lr,num_iter,device="cpu"):
    """To get the lr range
    """
    lr_finder = LRFinder(model,optimizer,criterion,device)
    lr_finder.range_test(train_loader,end_lr=end_lr,num_iter=num_iter,step_mode="exp")
    plot(lr_finder)
    lr_finder.reset()


def plot(
    lrfinder_class,
    skip_start=10,
    skip_end=5,
    log_lr=True,
    show_lr=None,
    ax=None,
    suggest_lr=True,
):
    """Plots the learning rate range test.

    Arguments:
        skip_start (int, optional): number of batches to trim from the start.
            Default: 10.
        skip_end (int, optional): number of batches to trim from the start.
            Default: 5.
        log_lr (bool, optional): True to plot the learning rate in a logarithmic
            scale; otherwise, plotted in a linear scale. Default: True.
        show_lr (float, optional): if set, adds a vertical line to visualize the
            specified learning rate. Default: None.
        ax (matplotlib.axes.Axes, optional): the plot is created in the specified
            matplotlib axes object and the figure is not be shown. If `None`, then
            the figure and axes object are created in this method and the figure is
            shown . Default: None.
        suggest_lr (bool, optional): suggest a learning rate by
            - 'steepest': the point with steepest gradient (minimal gradient)
            you can use that point as a first guess for an LR. Default: True.

    Returns:
        The matplotlib.axes.Axes object that contains the plot,
        and the suggested learning rate (if set suggest_lr=True).
    """

    if skip_start < 0:
        raise ValueError("skip_start cannot be negative")
    if skip_end < 0:
        raise ValueError("skip_end cannot be negative")
    if show_lr is not None and not isinstance(show_lr, float):
        raise ValueError("show_lr must be float")

    # Get the data to plot from the history dictionary. Also, handle skip_end=0
    # properly so the behaviour is the expected
    lrs = lrfinder_class.history["lr"]
    losses = lrfinder_class.history["loss"]
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    # Create the figure and axes object if axes was not already given
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # Plot loss as a function of the learning rate
    ax.plot(lrs, losses)

    # Plot the suggested LR
    if suggest_lr:
        # 'steepest': the point with steepest gradient (minimal gradient)
        print("LR suggestion: steepest gradient")
        min_grad_idx = None
        try:
            min_grad_idx = (np.gradient(np.array(losses))).argmin()
        except ValueError:
            print(
                "Failed to compute the gradients, there might not be enough points."
            )
        if min_grad_idx is not None:
            print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
            ax.scatter(
                lrs[min_grad_idx],
                losses[min_grad_idx],
                s=75,
                marker="o",
                color="red",
                zorder=3,
                label="steepest gradient",
            )
            ax.legend()

    if log_lr:
        ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")

    if show_lr is not None:
        ax.axvline(x=show_lr, color="red")

    # Show only if the figure was created internally
    if fig is not None:
        # plt.show()
        plt.savefig("temp.png")

    if suggest_lr and min_grad_idx is not None:
        return ax, lrs[min_grad_idx]
    else:
        return ax