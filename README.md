# 日本語テキスト匿名化ツール

企業文書や個人情報を含むテキストを安全に匿名化するためのPythonライブラリです。

## 特徴

- 高精度な日本語固有表現抽出
- 以下の情報を自動的に検出・匿名化
  - 人名
  - 会社名／組織名
  - 地名／住所
  - メールアドレス
  - 電話番号
  - 日付／時間
  - 金額
  - ID／ライセンス番号
  - プロジェクト名
  - 技術スタック
  - 部署名
- 検出したテキストの可逆的な匿名化（ハッシュ化）
- カスタマイズ可能なパターンマッチング
- 高速な処理と低メモリ使用

## 必要要件

- Python 3.8以上、3.12未満
- 必要なPythonパッケージ：
  - spacy~=3.5.0
  - fugashi~=1.2.0
  - unidic-lite~=1.0.8
  - intervaltree~=3.1.0
  - ipadic>=1.0.0
  - unidic>=1.1.0

## インストール方法

1. パッケージのインストール
```bash
# 基本機能のみをインストール
pip install japanese-text-anonymizer

# 開発環境込みでインストール（テストツール等を含む）
pip install japanese-text-anonymizer[dev]
```

2. 必要なモデルのダウンロード
```bash
# SpaCyの日本語モデルをインストール
python -m spacy download ja_core_news_lg

# または、セットアップスクリプトを使用
python scripts/setup.py
```

## 使用方法

### 基本的な使用例

```python
from japanese_text_anonymizer import JapaneseTextAnonymizer

# 匿名化ツールの初期化
anonymizer = JapaneseTextAnonymizer()

# テキストの匿名化
text = """
株式会社テクノソリューションズ（本社：東京都千代田区丸の内1-1-1）の
山田太郎部長は、新規プロジェクト「Project-X」の開始を発表しました。
連絡先：yamada.taro@example.com
"""

# 匿名化の実行
anonymized_text = anonymizer.anonymize_text(text)
print(anonymized_text)
# 出力例:
# <<a1b2c3d4>>（本社：<<e5f6g7h8>>）の
# <<i9j0k1l2>>は、新規プロジェクト「<<m3n4o5p6>>」の開始を発表しました。
# 連絡先：<<q7r8s9t0>>

# 必要に応じて元のテキストに戻す
original_text = anonymizer.decode_text(anonymized_text)
print(original_text)  # 元のテキストが表示されます
```

### カスタムパターンの追加

```python
# 既知のエンティティを追加
known_entities = {
    "Project-X": "PROJECT",
    "Docker": "TECH",
    "AWS": "TECH"
}

# テキストからパターンを学習
anonymizer.learn_from_text(text, known_entities)
```

### 統計情報の取得

```python
# 処理統計の取得
stats = anonymizer.get_statistics()
print(stats)
```

## 設定オプション

`JapaneseTextAnonymizer`クラスは以下のパラメータで初期化できます：

```python
anonymizer = JapaneseTextAnonymizer(
    min_confidence=0.3,  # エンティティ検出の最小信頼度
)
```

## 開発者向け情報

### 開発環境のセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/japanese-text-anonymizer.git
cd japanese-text-anonymizer

# 開発環境のインストール
pip install -e ".[dev]"

# セットアップスクリプトの実行
python scripts/setup.py
```

### テストの実行

```bash
# 全テストの実行
pytest

# カバレッジレポート付きでテストを実行
pytest --cov=japanese_text_anonymizer
```

### コードの品質管理

```bash
# コードフォーマットの確認
black .

# importの整理
isort .

# 型チェック
mypy .

# リンター実行
ruff check .
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 注意事項

- このツールは完全な匿名化を保証するものではありません。重要な個人情報の処理には十分な検証を行ってください。
- 大規模なテキストの処理には相応のメモリが必要となる場合があります。
- 匿名化されたテキストと元のテキストの対応関係は、プログラムの実行中のみメモリに保持されます。

## 貢献

バグ報告や機能要望は[Issue](https://github.com/raspverry/jp_hide_text/issues)にてお願いいたします。

プルリクエストも歓迎いたします。その際は以下の点にご注意ください：
- テストを追加してください
- コードスタイルを確認してください
- コミットメッセージは明確に記述してください

## サポート

ご不明な点がございましたら、[Issue](https://github.com/raspverry/jp_hide_text/issues)にてご質問ください。
