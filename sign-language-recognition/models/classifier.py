class SignLanguageClassifier:
    def __init__(self, model, input_shape, num_classes):
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_data, train_labels, validation_data, validation_labels, epochs=10, batch_size=32):
        history = self.model.fit(train_data, train_labels, 
                                 validation_data=(validation_data, validation_labels), 
                                 epochs=epochs, 
                                 batch_size=batch_size)
        return history

    def predict(self, input_data):
        predictions = self.model.predict(input_data)
        return predictions

    def evaluate(self, test_data, test_labels):
        evaluation = self.model.evaluate(test_data, test_labels)
        return evaluation

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)