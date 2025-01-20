# programming_contest_2024

高専プロコン2024競技部門用のコードです。

下記のpythonによるAI実装はあまりにも精度が振るわなかったので没にしました。
その代わりにnew_algoにc++メインの全探索系アルゴリズムを置いています。

/// new_algo ///
実行前にrequirements.txtを使用して仮想環境内にpythonライブラリをインストールしてください。

main.pyを実行すると結果が出力されるようになっていますが、main.py内にはサーバー通信用コードが入っているので別大会でも参考にしてもらえると幸いです。

/// old_algo ///

pytorchに関してはCUDAのバージョンでインストール方法が異なるので　[pytoch公式サイト](https://pytorch.org/get-started/locally/)　を参照してください。

run.pyを実行することで模擬環境で学習と評価を行うことができます。

run.py(newではmain.py)を動かした際に提出用のjsonファイルが作成されます
