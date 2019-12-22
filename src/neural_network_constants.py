class NNConstants(object):
    """neural network constants"""
    # neural network configurations
    num_features = 384
    hidden_layer_1_size = 256
    hidden_layer_2_size = 128

    # training configurations
    validation_split = 0.2
    batch_size = 64
    epochs = 50
    dropout_rate = 0.2
    learning_rate = 5e-5
