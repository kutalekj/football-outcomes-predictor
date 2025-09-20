import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

# Example data from your training log
rounds = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
val_accuracy = [0.7143, 0.625, 0.4444, 0.5333, 1.0, 0.4706, 0.6667, 0.2857, 0.7778, 0.3333, 0.5556, 0.5, 0.4444, 0.5333, 0.5000, 0.5556, 0.5000, 0.4444, 0.3333, 0.3333]
val_loss = [0.6369, 0.5959, 0.9408, 0.8299, 0.2340, 1.3475, 0.5027, 1.8884, 0.4887, 1.9981, 1.7126, 1.6938, 2.0818, 1.7933, 1.2790, 2.2205, 2.9168, 2.0300, 2.5814, 1.3457]

# Plot accuracy
fig, axs = plt.subplots(figsize=(10, 5))
axs.plot(rounds, val_accuracy, label="Validation Accuracy", marker="o")
axs.set_xlabel("Round")
axs.set_ylabel("Accuracy")
axs.set_title("Validation Accuracy Across Rounds")
axs.legend()
plt.show()

# Plot loss
fig, axs = plt.subplots(figsize=(10, 5))
axs.plot(rounds, val_loss, label="Validation Loss", marker="o", color='red')
axs.set_xlabel("Round")
axs.set_ylabel("Loss")
axs.set_title("Validation Loss Across Rounds")
axs.legend()
plt.show()

