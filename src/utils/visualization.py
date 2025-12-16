from matplotlib import pyplot as plt

def plot_predictions(train_data, val_data, test_data, train_pred_y, val_pred_y, test_pred_y):
    
    fig = plt.figure(figsize=(5, 6))
    train_pred_y = [y.cpu().numpy() for y in train_pred_y]
    plt.plot([1, 4.5], [1, 4.5], linestyle='--', c='k')
    plt.scatter([d.pl for d in train_data], train_pred_y, alpha=0.3, label ="tadf train")
    if val_pred_y is not None:
        val_pred_y = [y.cpu().numpy() for y in val_pred_y]
        plt.scatter([d.pl for d in val_data], val_pred_y, alpha=0.3, label='tadf val')
    if test_pred_y is not None:
        test_pred_y = [y.cpu().numpy() for y in test_pred_y]
        plt.scatter([d.pl for d in test_data], test_pred_y, alpha=0.3, label='tadf test')
    plt.xlim([1.5,4.5])
    plt.ylim([1.5,4.5])
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('TADF GAT Predictions')
    plt.grid()
    plt.show()
