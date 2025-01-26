import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=5.0):
        super(CustomLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, y_pred, y_true):
        # Loss Term: arctan or tanh for small-value sensitivity
        loss_term = 10 * torch.log(1 + torch.abs(y_pred - y_true))

        # Penalty Term: Large penalty for opposite signs
        sign_penalty = self.penalty_weight * (1 - torch.sign(y_pred * y_true)) * 0.01

        # Combine terms
        loss = loss_term + sign_penalty
        return torch.mean(loss)

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # Define the custom loss function
    def custom_loss(y_pred, y_true, penalty_weight=5.0):
        # Loss Term: arctan for small-value sensitivity
        loss_term = 10 * torch.log(1 + torch.abs(y_pred - y_true))
        # Penalty Term: Large penalty for opposite signs
        sign_penalty = penalty_weight * (1 - torch.sign(y_pred * y_true)) * 0.01
        # Combine terms
        return loss_term + sign_penalty


    # Generate data for graph
    y_pred = torch.linspace(-3, 3, 300)  # Predicted values
    y_true_positive = torch.tensor(1.0)  # Positive true value
    y_true_negative = torch.tensor(-1.0)  # Negative true value
    print(y_pred)
    print(y_true_positive)
    print(y_true_negative)
    # Compute loss for positive and negative true values
    loss_positive = custom_loss(y_pred, y_true_positive, penalty_weight=10.0).detach().numpy()
    loss_negative = custom_loss(y_pred, y_true_negative, penalty_weight=10.0).detach().numpy()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_pred.numpy(), loss_positive, label="True Value: +1.0", color="blue")
    # plt.plot(y_pred.numpy(), loss_negative, label="True Value: -1.0", color="red")
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Custom Loss Function Visualization")
    plt.xlabel("Predicted Value (y_pred)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
