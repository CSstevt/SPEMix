import torch
def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)['out']
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(accuracy)
    output_file = 'test_result.txt '
    with open(output_file, 'a') as file:
        file.write(f'Test Accuracy: {accuracy:.4f}\n')
    return accuracy

def val(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)['out']
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(accuracy)
    output_file = 'val_result.txt '
    with open(output_file, 'a') as file:
        file.write(f'Test Accuracy: {accuracy:.4f}\n')
    return accuracy

