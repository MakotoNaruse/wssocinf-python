from deep_convnet import DeepConvNet
from util import img_to_matrix


class ImageScore:
    def __init__(self):
        self.net = DeepConvNet()
        self.net.load_params('./neural_net/network_params.pkl')

    def _predict(self, file_path):
        x = img_to_matrix(file_path)
        return self.net.predict(x)

    def predict_score(self, file_path):
        x = self._predict(file_path)

        # 初期値(最低点)
        score = 50
        for i, xi in enumerate(x):
            score += xi * (i + 1) * (50 / 5)

        return score
