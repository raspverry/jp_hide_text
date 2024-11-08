"""
日本語テキスト匿名化ツールのセットアップスクリプト

このスクリプトは以下の処理を行います：
1. SpaCyの日本語モデルのインストール
2. 必要な依存パッケージの確認
"""

import subprocess
import sys
from typing import List, Optional

def setup_spacy_model() -> None:
    """SpaCyの日本語モデルをインストールする"""
    try:
        print("SpaCyの日本語モデルをインストールしています...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "spacy",
            "download",
            "ja_core_news_lg"
        ])
        print("日本語モデルのインストールが完了しました。")
    except subprocess.CalledProcessError as e:
        print(f"エラー: SpaCyモデルのインストールに失敗しました: {e}")
        sys.exit(1)

def verify_dependencies() -> Optional[List[str]]:
    """必要な依存パッケージが正しくインストールされているか確認する"""
    required_packages = [
        "spacy",
        "fugashi",
        "unidic-lite",
        "intervaltree"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages if missing_packages else None

def main() -> None:
    """セットアップのメイン処理"""
    print("セットアップを開始します...")
    
    # 依存パッケージの確認
    missing = verify_dependencies()
    if missing:
        print(f"警告: 以下のパッケージがインストールされていません：")
        for package in missing:
            print(f"- {package}")
        print("pip install -e '.[dev]' を実行してください。")
        sys.exit(1)
    
    # SpaCyモデルのインストール
    setup_spacy_model()
    
    print("セットアップが完了しました。")

if __name__ == "__main__":
    main()
