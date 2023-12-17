---
title: ローカルで動くLLMを試すためのスクリプト
date: 2023-09-20
lastmod: 2023-11-04
---

## 概要

## 実行方法

各スクリプトの実行方法は、スクリプトファイルの docstring に記載しています。
以下は、実行を想定しているスクリプトのファイル名と、スクリプトの簡易説明のみ記載します。
スクリプトのオプションは`python hoge.py -h`のようにしてオプションを出力して確認してください。

- elyza.py: [ELYZA のモデル](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruce)を利用した文章生成スクリプト。
- llm-jp.py: [llm-jp](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0)を利用した文章生成スクリプト。モデルの読み込みに非常に時間がかかった。

## 環境構築

```sh
# torch cu117版を指定してますが、環境に合わせて適切なバージョンを指定してください。
# 動作確認したバージョンを固定で導入する場合
$ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
# or
# 最新のバージョンを確認して導入する場合
$ pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
```

### 開発環境構築

```sh
# 動作確認したバージョンを固定で導入する場合
$ pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu117
# or
# 最新のバージョンを確認して導入する場合
$ pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu117
```

### 実行環境の更新

既存の venv 環境を削除後に下記のコマンドで環境を構築する。

```sh
$ pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
$ pip freeze > requirements.txt
# requirements.txtに対して下記の変更を実施
#
# - pytorchのcudaバージョン指定を削除
# - `-e`で指定されている行を削除

# 開発環境の構築
# `-c`オプションでrequirements.txtの内容は一致させる
$ pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu117
$ pip freeze > requirements-dev.txt  # requirements.txtと同様に処理
```

## Tips
