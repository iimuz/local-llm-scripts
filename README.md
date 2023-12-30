---
title: ローカルで動くLLMを試すためのスクリプト
date: 2023-09-20
lastmod: 2023-12-30
---

## 概要

## 実行方法

各スクリプトの実行方法は、スクリプトファイルの docstring に記載しています。
以下は、実行を想定しているスクリプトのファイル名と、スクリプトの簡易説明のみ記載します。
スクリプトのオプションは`python hoge.py -h`のようにしてオプションを出力して確認してください。

- elyza.py: [ELYZA のモデル](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruce)を利用した文章生成スクリプト。
- llm-jp.py: [llm-jp](https://huggingface.co/llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0)を利用した文章生成スクリプト。モデルの読み込みに非常に時間がかかった。

## 環境構築

事前に下記が利用できるように環境を設定してください。

- [node.js](https://nodejs.org/en)
- [python](https://nodejs.org/en)
- [task](https://taskfile.dev/): タスクランナーとして利用します。

仮想環境などの構築は下記のコマンドで実行します。

```sh
# 実行だけできればよい場合
task init
# 開発環境もインストールする場合
task init-dev
```

## Tips

### タスク一覧

実行可能なタスク一覧は下記のコマンドで確認してください。

```sh
task -l
```
