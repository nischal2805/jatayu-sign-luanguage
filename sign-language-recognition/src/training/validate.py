def validate_model(model, validation_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(validation_loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

def main():
    import torch
    from src.config import Config
    from src.models.classifier import SignLanguageClassifier
    from src.data_processing.preprocessor import get_validation_loader

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignLanguageClassifier()
    model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    validation_loader = get_validation_loader(config.validation_data_path)

    criterion = torch.nn.CrossEntropyLoss()
    average_loss, accuracy = validate_model(model, validation_loader, criterion, device)

    print(f'Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()