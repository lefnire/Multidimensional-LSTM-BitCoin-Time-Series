import time, pdb
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# If a plot exists, we'll update it
Existing = namedtuple('Existing', ['li_true', 'li_pred', 'fig'])
existing = None


def plot_results(predicted_data, true_data, block=False):
    global Existing, existing

    if existing:
        print('existing')
        existing.li_true.set_ydata(true_data)
        existing.li_pred.set_ydata(predicted_data)
        existing.fig.canvas.draw()
    else:
        print('not existing')
        fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        li_true, = ax.plot(true_data, label='True Data')
        li_pred, = plt.plot(predicted_data, label='Prediction')
        plt.legend()
        plt.show(block=block)
        existing = Existing(li_true=li_true, li_pred=li_pred, fig=fig)


def realtime_plot(): # TODO
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # some X and Y data
    x = np.arange(10000)
    y = np.random.randn(10000)

    li, = ax.plot(x, y)

    # draw and show it
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()
    plt.show(block=False)

    # loop to update the data
    while True:
        try:
            y[:-10] = y[10:]
            y[-10:] = np.random.randn(10)
            li.set_ydata(y) # set the new data
            fig.canvas.draw()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()