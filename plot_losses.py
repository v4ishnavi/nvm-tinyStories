import re
import argparse
import matplotlib.pyplot as plt


def parse_losses(file_path):
    # Initialize lists to store extracted losses
    training_losses = []
    validation_losses = []
    training_batch_losses = []
    validation_batch_losses = []

    # Compile regular expressions to match the patterns
    training_loss_pattern = re.compile(r'Training Losses: \[(.*?)\]')
    validation_loss_pattern = re.compile(r'Validation Losses: \[(.*?)\]')
    training_batch_loss_pattern = re.compile(r'Training Batch Losses: \[(.*?)\]')
    validation_batch_loss_pattern = re.compile(r'Validation Batch Losses: \[(.*?)\]')

    # Read the log file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the patterns and extract the losses
            training_loss_match = re.search(training_loss_pattern, line)
            validation_loss_match = re.search(validation_loss_pattern, line)
            training_batch_loss_match = re.search(training_batch_loss_pattern, line)
            validation_batch_loss_match = re.search(validation_batch_loss_pattern, line)

            # If a match is found, convert the string of numbers to a list of floats
            if training_loss_match:
                training_losses = list(map(float, training_loss_match.group(1).split(',')))
            if validation_loss_match:
                validation_losses = list(map(float, validation_loss_match.group(1).split(',')))
            if training_batch_loss_match:
                training_batch_losses = list(map(float, training_batch_loss_match.group(1).split(',')))
            if validation_batch_loss_match:
                validation_batch_losses = list(map(float, validation_batch_loss_match.group(1).split(',')))

    return training_losses, validation_losses, training_batch_losses, validation_batch_losses


def plot_losses(training_losses, output_path, title):
    epochs = range(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses)

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Save the plot to the specified file path
    plt.savefig(output_path)
    plt.close()


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Parse log file and plot training/validation losses.')
    parser.add_argument('log_file', type=str, help='Path to the log file to parse.')
    parser.add_argument('output_path_train', type=str, help='Path to save the plot image.')
    parser.add_argument('output_path_batch_train', type=str, help='Path to save the plot image for batch-wise loss.')
    parser.add_argument('output_path_validate', type=str, help='Path to save the plot image.')
    parser.add_argument('output_path_batch_validate', type=str, help='Path to save the plot image for batch-wise loss.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Parse the log file
    training_losses, validation_losses, training_batch_losses, validation_batch_losses = parse_losses(args.log_file)

    # Plot the training and validation losses
    plot_losses(training_losses, args.output_path_train, "Training Loss (per epoch)")
    plot_losses(validation_losses, args.output_path_validate, "Validation Loss (per epoch)")
    plot_losses(training_batch_losses, args.output_path_batch_train, "Training Loss (per batch)")
    plot_losses(validation_batch_losses, args.output_path_batch_validate, "Validation Loss (per batch)")


if __name__ == '__main__':
    main()
