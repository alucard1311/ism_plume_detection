#!/usr/bin/env python3
import numpy as np
import rospy
from plume_detection.msg import Hist  # Adjust with your actual package name
import matplotlib.pyplot as plt
import threading

# Global variables to store the latest histogram data
latest_bins = None
latest_values = None
data_ready = threading.Event()

def histogram_callback(data):
    global latest_bins, latest_values, data_ready
    latest_bins = data.bins
    latest_values = data.values
    data_ready.set()  # Signal that new data is ready to be plotted

def plot_histogram_with_entropy(bins, values):
    # Normalize the histogram values to probabilities
    total = sum(values)
    probabilities = [v / total for v in values if v > 0]

    # Calculate the entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    # Update the histogram plot
    plt.clf()
    plt.bar(bins, values, width=0.8, color='blue')
    plt.xlabel('Bins')
    plt.ylabel('Values')
    plt.title(f'Histogram with Entropy: {entropy:.2f} bits')
    plt.pause(0.1)  # Pause to allow the plot to update

def listener():
    rospy.init_node('histogram_listener', anonymous=True)
    rospy.Subscriber('mean_histogram', Hist, histogram_callback)
    plt.ion()  # Turn the interactive mode on
    plt.show()

    while not rospy.is_shutdown():
        data_ready.wait()  # Wait for the signal that new data is ready
        if latest_bins is not None and latest_values is not None:
            plot_histogram_with_entropy(latest_bins, latest_values)
        data_ready.clear()  # Reset the event until new data arrives

if __name__ == '__main__':
    listener()
