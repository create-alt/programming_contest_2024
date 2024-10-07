# programming_contest_2024

高専プロコン2024競技部門用のコードです。

実行前にrequirements.txtを使用して仮想環境内にpythonライブラリをインストールしてください。

pytorchに関してはCUDAのバージョンでインストール方法が異なるので　[pytoch公式サイト](https://pytorch.org/get-started/locally/)　を参照してください。

run.pyを実行することで模擬環境で学習と評価を行うことができます。

現在はまだできていませんがEnvironment.cppをmodel.py内で読み込み、環境(transition class)をcppで動かすことで実行時間の短縮を図ります。
（pythonのみで高速化させることができたので必要なし）

run.pyを動かした際に提出用のjsonファイルが作成されます

このjsonファイルの生成やmodelの重み保存のために実行するときには cd runs でカレントディレクトリを変更してから実行してください。

## 現在実装できている機能
ボード・目標・抜き型を与えた際に学習（現在少ししか進んでいないので改善必須）
 
行動、行動後のボードの描画
  
提出用jsonファイルの作成

サーバーとの通信用コード

run.py中での入力jsonファイルの使用

## 未実装の機能
