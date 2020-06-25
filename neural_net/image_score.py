from deep_convnet import DeepConvNet
from util import im2mat
import numpy as np


class ImageScore:
    def __init__(self, input_dim=(3, 32, 32)):
        self.image_size = input_dim[1:]
        self.net = DeepConvNet(input_dim)
        self.net.load_params('./neural_net/network_params.pkl')

    def _predict(self, file_path):
        x = im2mat(file_path, self.image_size)
        return self.net.predict(np.expand_dims(x, 0))

    def predict_score(self, file_path):
        y = self._predict(file_path)
        y = np.squeeze(y)

        score = 50
        for i, yi in enumerate(y):
            score += yi * (i + 1) * (50 / 5)

        return score


# test code
def main(file_path):
    ims = ImageScore()
    print(ims.predict_score(file_path))


if __name__ == "__main__":
    main('./neural_net/test.jpg')
