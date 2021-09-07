import numpy as np
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
from torch import optim
from sklearn.model_selection import train_test_split





# 手書き数字の画像データMNISTをダウンロード
mnist = fetch_openml('mnist_784', version=1, data_home=".")



# 1. データの前処理（画像データとラベルに分割し、正規化）
X = mnist.data / 255  # 0-255を0-1に正規化
y = mnist.target

y = np.array(y)
y = y.astype(np.int32)



# 2.1 データを訓練とテストに分割（6:1）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/7, random_state=0)


# 2.2 データをPyTorchのTensorに変換
X_train = torch.Tensor(X_train.values)
X_test = torch.Tensor(X_test.values)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# 2.3 データとラベルをセットにしたDatasetを作成
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)


# 2.4 データセットのミニバッチサイズを指定した、Dataloaderを作成
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)



# 3. ネットワークの構築
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))



# 4. 誤差関数と最適化手法の設定
# 誤差関数の設定
loss_fn = nn.CrossEntropyLoss()  # 変数名にはcriterionが使われることも多い

# 重みを学習する際の最適化手法の選択
optimizer = optim.Adam(model.parameters(), lr=0.01)



# 5. 学習と推論の設定
# 5-1. 学習1回でやることを定義します
# Chainerのtraining.Trainer()に対応するものはない


def train(epoch):
    model.train()  # ネットワークを学習モードに切り替える

    # データローダーから1ミニバッチずつ取り出して計算する
    for data, targets in loader_train:
      
        optimizer.zero_grad()  # 一度計算された勾配結果を0にリセット
        outputs = model(data)  # 入力dataをinputし、出力を求める
        loss = loss_fn(outputs, targets)  # 出力と訓練データの正解との誤差を求める
        loss.backward()  # 誤差のバックプロパゲーションを求める
        optimizer.step()  # バックプロパゲーションの値で重みを更新する

    print("epoch{}：終了\n".format(epoch))


# 5-2. 推論1回でやることを定義
# Chainerのtrainer.extend(extensions.Evaluator())に対応するものはない


def test():
    model.eval()  # ネットワークを推論モードに切り替える
    correct = 0

    # データローダーから1ミニバッチずつ取り出して計算する
    with torch.no_grad():  # 微分は推論では必要ない
        for data, targets in loader_test:

            outputs = model(data)  # 入力dataをinputし、出力を求める

            # 推論する
            _, predicted = torch.max(outputs.data, 1)  # 確率が最大のラベルを求める
            correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 正解と一緒だったらカウントアップ

    # 正解率を出力
    data_num = len(loader_test.dataset)  # データの総数
    print('\nテストデータの正解率: {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))


# 学習なしにテストデータで推論
test()



# 6. 学習と推論の実行
for epoch in range(3):
    train(epoch)

test()