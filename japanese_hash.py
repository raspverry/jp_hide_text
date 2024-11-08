import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import fugashi
import spacy
from spacy.tokens import Doc, Token


@dataclass
class EntityPattern:
    """エンティティパターン情報を保存するクラス"""

    text: str
    label: str
    pos: str
    features: Dict[str, str]
    context: List[str]


class JapaneseTextAnonymizer:
    def __init__(self):
        self.nlp = spacy.load("ja_core_news_lg")
        self.tagger = fugashi.Tagger()

        # エンティティタイプ定義
        self.target_entities = {
            # 既存のエンティティ
            "ORG",  # 会社/組織名
            "PERSON",  # 名前
            "LOC",  # 地名
            "GPE",  # 国/都市名
            "PRODUCT",  # 製品名
            "FACILITY",  # 施設名
            "EMAIL",  # メールアドレス
            "PHONE",  # 電話番号
            "URL",  # URL
            "DATE",  # 日付
            "TIME",  # 時間
            "MONEY",  # 金額
            "PERCENT",  # ％
            "ID",  # 識別子(社員番号など)
            "LICENSE",  # ライセンス/資格証
            "TECH",  # 技術スタック
            "PROJECT",  # プロジェクト名
            "DEPT",  # 部署名
            "BRANCH",  # 支店名
            "SYSTEM",  # システム名
            "DURATION",  # 期間
        }

        # 肩書きパターン定義
        self.title_patterns = {
            # 一般の肩書き
            "部長",
            "課長",
            "係長",
            "主任",
            "担当",
            "社長",
            "副社長",
            "専務",
            "常務",
            "取締役",
            # 職位/職責
            "代表取締役",
            "執行役員",
            "理事",
            "監査役",
            "事業部長",
            "本部長",
            "支店長",
            "営業部長",
            "技術部長",
            "開発部長",
            "総務部長",
            "人事部長",
            "マネージャー",
            "リーダー",
            "チームリーダー",
            "グループリーダー",
            "プロジェクトリーダー",
            # 学術/教育関連
            "教授",
            "准教授",
            "講師",
            "助教",
            "研究員",
            "博士",
            "修士",
            "学士",
            "先生",
            "教諭",
            # 尊称
            "氏",
            "様",
            "さん",
            "君",
            "殿",
            "先輩",
            "後輩",
            "師匠",
            # 技術の肩書き
            "エンジニア",
            "プログラマー",
            "デザイナー",
            "アーキテクト",
            "コンサルタント",
            "プロジェクトマネージャー",
            "プロダクトマネージャー",
            "テクニカルリード",
            "スクラムマスター",
            # 一般職務
            "会計士",
            "弁護士",
            "税理士",
            "司法書士",
            "医師",
            "看護師",
            "薬剤師",
            # 略語および英語の肩書き
            "Sr",
            "Jr",
            "CEO",
            "CTO",
            "CFO",
            "COO",
            "VP",
            "SVP",
            "EVP",
            "GM",
            "PM",
            "TL",
            # 接尾辞
            "級",
            "等",
            "世",
        }

        # 各肩書きに対する正規表現式追加
        self.title_patterns_regex = set()
        for title in self.title_patterns:
            # 括弧内の肩書きも含む
            self.title_patterns_regex.add(rf"\({title}\)")
            # 英語の肩書き後の、コンマを含む
            if all(ord(c) < 128 for c in title):  # ASCII文字のみの場合
                self.title_patterns_regex.add(rf"{title},")
        self.title_patterns_regex.update(self.title_patterns)

        # エンティティ別正規式パターン
        self.entity_patterns = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": [
                r"(?:\+81|0)\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}",
                r"\(\d{2,4}\)\d{2,4}-\d{4}",
            ],
            "URL": r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+",
            "DATE": [
                r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?",
                r"令和\d{1,2}年\d{1,2}月\d{1,2}日",
                r"平成\d{1,2}年\d{1,2}月\d{1,2}日",
            ],
            "TIME": [
                r"\d{1,2}[:時]\d{2}分?",
                r"\d{1,2}時\d{2}分",
            ],
            "MONEY": [
                r"¥?\d{1,3}(?:,\d{3})*(?:\.\d{2})?円?",  # 基本金額
                r"(?:約|およそ|概算で?)?[\s\t]*¥?\d{1,3}(?:,\d{3})*(?:\.\d{2})?円?[\s\t]*(?:程度|ほど|前後|以上|以下|未満|超)?",
                r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:円|万円|億円|兆円)",
            ],
            "PERCENT": r"\d+(?:\.\d+)?%",
            "ID": [
                r"[A-Z]{2,3}\d{5,}",  # 例: EMP12345
                r"ID[-:]?\d{5,}",  # 例 ID-12345
            ],
            "LICENSE": [
                r"[第]?\d{5,}号?",  # 例 第12345号
                r"[A-Z]\d{4,}",  # 例 A12345
            ],
            "DURATION": [
                r"\d+年(?:\d+ヶ月|\d+か月)?",
                r"\d+ヶ月|\d+か月",
                r"\d+日間?",
            ],
        }

        # プロジェクト名パターン
        self.project_patterns = {
            r"Project[-\s]?[A-Z0-9]+",  # Project-X, Project A
            r"[A-Z][a-z]+[-\s]?[0-9]+",  # Phase-1, Version 2
            r"[A-Z][a-z]+\s+[A-Z][a-z]+",  # Blue Ocean
            r"[\(（][Pp]roject\s*[:：]?\s*[^\)）]+[\)）]",  # (Project: NAME)
            r"「[Pp]roject\s*[:：]?\s*[^」]+」",  # 「Project: NAME」
            r"[!\w\s-]{3,}[Pp]roject",  # Something Project
            r"[Pp]roject[\s\w-]{3,}",  # Project Something
        }

        # 技術スタック関連パターン
        self.tech_patterns = {
            # プログラミング言語
            r"Python",
            r"Java",
            r"JavaScript",
            r"TypeScript",
            r"C\+\+",
            r"C#",
            r"Ruby",
            r"PHP",
            r"Go",
            r"Rust",
            r"Swift",
            r"Kotlin",
            r"Scala",
            r"R",
            r"MATLAB",
            r"Perl",
            r"Shell",
            r"Bash",
            r"PowerShell",
            # フレームワーク/ライブラリ
            r"Django",
            r"Flask",
            r"FastAPI",
            r"Spring",
            r"Spring Boot",
            r"React",
            r"Vue",
            r"Angular",
            r"Svelte",
            r"Next\.js",
            r"Nuxt\.js",
            r"TensorFlow",
            r"PyTorch",
            r"Keras",
            r"scikit-learn",
            r"Pandas",
            r"NumPy",
            r"SciPy",
            r"Matplotlib",
            r"Express",
            r"Nest\.js",
            r"Laravel",
            r"Ruby on Rails",
            # 데이터베이스
            r"MySQL",
            r"PostgreSQL",
            r"MongoDB",
            r"Redis",
            r"Elasticsearch",
            r"Oracle",
            r"SQL Server",
            r"SQLite",
            r"MariaDB",
            r"DynamoDB",
            r"Cassandra",
            r"Neo4j",
            r"InfluxDB",
            # クラウド/インフラ
            r"AWS",
            r"Amazon Web Services",
            r"Azure",
            r"GCP",
            r"Google Cloud Platform",
            r"Heroku",
            r"DigitalOcean",
            r"Docker",
            r"Kubernetes",
            r"K8s",
            r"OpenShift",
            r"Terraform",
            r"Ansible",
            r"Chef",
            r"Puppet",
            # ツール/サービス
            r"Git",
            r"GitHub",
            r"GitLab",
            r"Bitbucket",
            r"Jenkins",
            r"CircleCI",
            r"Travis CI",
            r"ArgoCD",
            r"Prometheus",
            r"Grafana",
            r"ELK Stack",
            r"Datadog",
            r"New Relic",
            # プロトコル/標準
            r"REST",
            r"RESTful",
            r"GraphQL",
            r"gRPC",
            r"WebSocket",
            r"OAuth",
            r"JWT",
            r"SOAP",
            r"HTTP/[0-9.]+",
        }

        # 部署パターンの改善
        self.dept_full_patterns = {
            "prefix": {
                "第一",
                "第二",
                "第三",
                "第四",
                "第五",
                "グローバル",
                "海外",
                "国内",
                "統合",
                "事業",
                "製品",
                "サービス",
                "ソリューション",
                "コーポレート",
                "総合",
                "先端",
                "次世代",
            },
            "core": {
                "開発",
                "技術",
                "営業",
                "総務",
                "人事",
                "経理",
                "企画",
                "広報",
                "法務",
                "調査",
                "研究",
                "品質管理",
                "カスタマーサービス",
                "マーケティング",
                "セールス",
                "情報システム",
                "デジタル",
                "イノベーション",
                "インフラ",
                "セキュリティ",
                "データ",
                "AI",
            },
            "suffix": {
                "部",
                "課",
                "室",
                "チーム",
                "グループ",
                "事業部",
                "本部",
                "センター",
                "ユニット",
                "推進室",
                "統括部",
                "支援室",
            },
        }

        # 住所の構成要素の拡張
        self.address_units = {
            # 基本単位
            "都",
            "道",
            "府",
            "県",
            "市",
            "区",
            "町",
            "村",
            # 細部の住所
            "丁目",
            "番地",
            "号",
            # 建物関連
            "ビル",
            "タワー",
            "マンション",
            "ハイツ",
            "コーポ",
            "パーク",
            # 方向
            "東",
            "西",
            "南",
            "北",
            # その他
            "街区",
            "団地",
            "アパート",
        }

        # キャッシュおよび設定
        self.learned_patterns: Dict[str, List[EntityPattern]] = defaultdict(list)
        self.entity_texts: Set[str] = set()
        self.hash_cache: Dict[str, str] = {}
        self.reverse_cache: Dict[str, str] = {}
        self.processed_cache: Dict[str, List[Tuple[int, int, str]]] = {}
        self.min_confidence = 0.3

    def _generate_hash(self, text: str) -> str:
        """テキストをSHA-256ハッシュに変換"""
        if text in self.hash_cache:
            return self.hash_cache[text]

        hash_obj = hashlib.sha256(text.encode("utf-8"))
        hashed = f"<<{hash_obj.hexdigest()[:8]}>>"
        self.hash_cache[text] = hashed
        self.reverse_cache[hashed] = text
        return hashed

    def _extract_features(self, token: Token) -> Dict[str, str]:
        """トークンの形態素の特性抽出"""
        return {
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "shape": token.shape_,
            "is_alpha": str(token.is_alpha),
            "is_ascii": str(token.is_ascii),
            "is_digit": str(token.is_digit),
            "like_num": str(token.like_num),
            "is_title": str(token.is_title),
            "is_bracket": str(
                bool(re.match(r"[\(\［\｢\「].*?[\)\］\｣\」]", token.text))
            ),
            "has_number": str(bool(re.search(r"\d", token.text))),
            "length": str(len(token.text)),
        }

    def _get_context(
        self, doc: Doc, start_idx: int, end_idx: int, window: int = 2
    ) -> List[str]:
        """始まりと終わりのインデックスを基盤に文脈抽出"""
        start = max(0, start_idx - window)
        end = min(len(doc), end_idx + window)

        context = []
        for i in range(start, end):
            if i < start_idx or i >= end_idx:
                context.append(doc[i].text)
        return context

    def _process_identifier(
        self, text: str, pattern: str
    ) -> List[Tuple[int, int, str]]:
        """識別子(ID、ライセンス番号など)処理"""
        results = []
        processed_spans = set()

        id_pattern = re.compile(
            r"(?P<prefix>[第])?" r"(?P<number>\d+)" r"(?P<suffix>号)?"
        )

        for match in id_pattern.finditer(text):
            start, end = match.span()
            if any(
                p_start <= start < p_end or p_start < end <= p_end
                for p_start, p_end in processed_spans
            ):
                continue

            groups = match.groupdict()
            result = ""

            if groups["prefix"]:
                result += groups["prefix"]

            # 숫자 부분만 해시화
            number_hash = self._generate_hash(groups["number"])
            result += number_hash

            if groups["suffix"]:
                result += groups["suffix"]

            results.append((start, end, result))
            processed_spans.add((start, end))

        return sorted(results, key=lambda x: x[0])

    def _process_address_numbers(self, text: str) -> List[Tuple[int, int, str]]:
        """住所の番地数処理"""
        results = []
        # 日本式の住所番号パターン (例 1-1-1, 1番地1号)
        patterns = [
            r"\d+(?:[-−]\d+){1,2}",  # 1-1-1 形式
            r"\d+番地?\d*号?",  # 1番地1号 形式
            r"\d+(?:丁目|番|号)",  # 1丁目, 1番, 1号 形式
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                results.append(
                    (match.start(), match.end(), self._generate_hash(match.group()))
                )

        return results

    def _split_address(self, text: str) -> List[Tuple[int, int, str]]:
        """住所を構成要素別に分離"""
        results = []
        processed_spans = set()

        # 1. 番地パターン処理
        number_patterns = [
            r"\d+(?:[-−]\d+){1,2}",  # 1-1-1 形式
            r"\d+番地?\d*号?",  # 1番地1号 形式
            r"\d+(?:丁目|番|号)",  # 1丁目, 1番, 1号 形式
            r"[一二三四五六七八九十百]+(?:番地?|号)",  # 漢字・数字
        ]

        for pattern in number_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                if not any(
                    (s <= start < e or s < end <= e) for s, e in processed_spans
                ):
                    processed_spans.add((start, end))
                    results.append((start, end, self._generate_hash(match.group())))

        # 2.アドレス単位処理
        current_start = 0
        current_text = ""
        i = 0

        while i < len(text):
            if any((start <= i < end) for start, end in processed_spans):
                i += 1
                continue

            current_text += text[i]
            matching_units = [
                unit for unit in self.address_units if current_text.endswith(unit)
            ]

            if matching_units:
                matched_unit = max(matching_units, key=len)
                if current_text.strip():
                    start = i + 1 - len(current_text)
                    end = i + 1
                    if not any(
                        (s <= start < e or s < end <= e) for s, e in processed_spans
                    ):
                        processed_spans.add((start, end))
                        results.append((start, end, self._generate_hash(current_text)))
                current_text = ""
                current_start = i + 1
            i += 1

        # 3. 残ったテキスト処理
        if current_text.strip():
            start = len(text) - len(current_text)
            end = len(text)
            if not any((s <= start < e or s < end <= e) for s, e in processed_spans):
                results.append((start, end, self._generate_hash(current_text)))

        return sorted(results, key=lambda x: x[0])

    def _find_tech_stack(self, text: str) -> List[Tuple[int, int, str]]:
        """技術スタック関連用語の検索"""
        results = []
        processed_positions = set()

        # 複合キスルミョンのための特別パターン
        compound_tech_patterns = {
            r"Ruby\s+on\s+Rails",
            r"Microsoft\s+Azure",
            r"Google\s+Cloud\s+Platform",
            r"Amazon\s+Web\s+Services",
            r"Visual\s+Studio\s+Code",
            r"Amazon\s+EC2",
            r"Google\s+Cloud",
        }

        # 1. まず、複合パターン処理
        for pattern in sorted(compound_tech_patterns, key=len, reverse=True):
            regex = rf"\b{pattern}\b"
            for match in re.finditer(regex, text, re.IGNORECASE):
                start, end = match.span()
                if not any(
                    p_start <= start < p_end or p_start < end <= p_end
                    for p_start, p_end in processed_positions
                ):
                    processed_positions.add((start, end))
                    results.append((start, end, self._generate_hash(match.group())))

        # 2. 単一キスルミョン処理(警戒条件の強化)
        tech_patterns_with_boundaries = {
            rf"\b{pattern}\b" if not pattern.startswith(r"\b") else pattern
            for pattern in self.tech_patterns
        }

        # 正確なマッチングを向け、パターン整列
        sorted_patterns = sorted(tech_patterns_with_boundaries, key=len, reverse=True)

        for pattern in sorted_patterns:
            # 日本語文字の前後の境界も考慮
            pattern = f"(?:^|[^ぁ-んァ-ン一-龯]){pattern}(?:$|[^ぁ-んァ-ン一-龯])"

            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 실私がマッチングした部分から前後の文字を除外
                full_match = match.group()
                if full_match[0].isspace() or full_match[0] in ".,;:":
                    start = match.start() + 1
                else:
                    start = match.start()

                if full_match[-1].isspace() or full_match[-1] in ".,;:":
                    end = match.end() - 1
                else:
                    end = match.end()

                if not any(
                    p_start <= start < p_end or p_start < end <= p_end
                    for p_start, p_end in processed_positions
                ):
                    processed_positions.add((start, end))
                    # 元の大文字と小文字の保持
                    original_text = text[start:end]
                    results.append((start, end, self._generate_hash(original_text)))

        return sorted(results, key=lambda x: x[0])

    def _process_money(self, text: str) -> List[Tuple[int, int, str]]:
        """金額表現処理"""
        results = []
        processed_spans = set()

        # 金額パターン定義
        money_pattern = re.compile(
            r"(?P<prefix>約|およそ|概算で)?"
            r"(?P<currency>[¥￥])?"
            r"(?P<amount>(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)"
            r"(?P<unit>兆円|億円|万円|円)?"
            r"(?P<suffix>程度|ほど|前後|以上|以下|未満|超)?"
        )

        for match in money_pattern.finditer(text):
            start, end = match.span()
            if any(
                p_start <= start < p_end or p_start < end <= p_end
                for p_start, p_end in processed_spans
            ):
                continue

            groups = match.groupdict()
            if not groups["amount"]:
                continue

            # 元の金額を保存
            original_amount = groups["amount"]

            # ハッシュ生成(コンマ除去)
            pure_amount = original_amount.replace(",", "")
            hashed_amount = self._generate_hash(pure_amount)

            # フォーマット復元
            formatted_hash = self._restore_format(original_amount, hashed_amount)

            # 最終結果の組み合わせ
            result = ""
            if groups["prefix"]:
                result += groups["prefix"]
            if groups["currency"]:
                result += groups["currency"]

            result += formatted_hash

            if groups["unit"]:
                result += groups["unit"]
            if groups["suffix"]:
                result += groups["suffix"]

            results.append((start, end, result))
            processed_spans.add((start, end))

        return sorted(results, key=lambda x: x[0])

    def _restore_format(self, amount: str, hashed: str) -> str:
        """金額の原本フォーマット(コンマなど)の復元"""
        # 元の金額からコンマの位置を探す
        comma_positions = [i for i, char in enumerate(amount) if char == ","]
        if not comma_positions:
            return hashed

        # ハッシュされた数字にコンマを挿入
        result = list(hashed.replace(",", ""))
        offset = 0
        for pos in comma_positions:
            if pos + offset < len(result):
                result.insert(pos + offset, ",")
                offset += 1

        return "".join(result)

    def _find_department(self, text: str) -> List[Tuple[int, int, str]]:
        """改善された部署名を探す"""
        results = []
        processed_positions = set()

        # 全体部署名パターンマッチング
        patterns = []
        # 基本パターン (prefix? + core + suffix?)
        for core in self.dept_full_patterns["core"]:
            for suffix in self.dept_full_patterns["suffix"]:
                for prefix in self.dept_full_patterns["prefix"]:
                    patterns.append(f"{prefix}?{core}{suffix}")
                patterns.append(f"{core}{suffix}")

        # 整列されたパターン(長いものからマッチング)
        sorted_patterns = sorted(patterns, key=len, reverse=True)

        for pattern in sorted_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()

                # 重複チェック
                overlap = False
                for pos_start, pos_end in processed_positions:
                    if (start >= pos_start and start < pos_end) or (
                        end > pos_start and end <= pos_end
                    ):
                        overlap = True
                        break

                if not overlap:
                    processed_positions.add((start, end))
                    results.append((start, end, self._generate_hash(match.group())))

        return results

    def _find_projects(self, text: str) -> List[Tuple[int, int, str]]:
        """プロジェクト名を探す"""
        results = []
        processed_positions = set()

        # プロジェクト名パターン定義
        full_patterns = [
            r"「[^」]*?[Pp]roject[-\s]?[A-Z0-9]+[^」]*?」",  # 「Project-X」形式
            r"『[^』]*?[Pp]roject[-\s]?[A-Z0-9]+[^』]*?』",  # 『Project-X』形式
            r"\([^)]*?[Pp]roject[-\s]?[A-Z0-9]+[^)]*?\)",  # (Project-X)形式
            r"［[^］]*?[Pp]roject[-\s]?[A-Z0-9]+[^］]*?］",  # ［Project-X］形式
            r"[Pp]roject[-\s]?[A-Z0-9][-_\s]*[A-Za-z0-9]+",  # Project-X Plus 形式
            r"[A-Z][a-z]+[-\s]?[Pp]roject[-\s]?[A-Z0-9]+",  # Alpha Project-1 形式
            r"[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[Pp]roject)?",  # Blue Ocean Project 形式
        ]

        # 1. 全体プロジェクト名マッチング
        for pattern in full_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                if not any(
                    (s <= start < e or s < end <= e) for s, e in processed_positions
                ):
                    processed_positions.add((start, end))
                    results.append((start, end, self._generate_hash(match.group())))

        return sorted(results, key=lambda x: x[0])

    def _has_title_suffix(self, text: str) -> Tuple[bool, str, str]:
        """テキストで肩書接尾辞を探して分離"""
        for pattern in sorted(self.title_patterns_regex, key=len, reverse=True):
            if text.endswith(pattern):
                return True, text[: -len(pattern)], pattern
        return False, text, ""

    def _is_span_available(
        self, processed_spans: Set[Tuple[int, int]], start: int, end: int
    ) -> bool:
        """与えられた範囲が既に処理された範囲と重ならないことを確認"""
        for proc_start, proc_end in processed_spans:
            if not (end <= proc_start or start >= proc_end):
                return False
        return True

    def _get_non_overlapping_replacements(
        self, replacements: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """重ならない交換項目を選択"""
        if not replacements:
            return []

        # スタート位置に整列して、同じスタート位置の場合、さらに長い区間まず
        sorted_replacements = sorted(replacements, key=lambda x: (x[0], -x[1]))
        result = []
        current = sorted_replacements[0]

        for next_item in sorted_replacements[1:]:
            # 重ならない場合
            if next_item[0] >= current[1]:
                result.append(current)
                current = next_item
            # 重なる場合、もっと長い区間を選択
            elif next_item[1] - next_item[0] > current[1] - current[0]:
                current = next_item

        result.append(current)
        return sorted(result, key=lambda x: x[0], reverse=True)

    def _find_regex_entities(
        self, text: str, pattern: str, label: str
    ) -> List[Tuple[int, int, str]]:
        """正規式のパターンにエンティティを探す"""
        results = []
        for match in re.finditer(pattern, text):
            self.entity_texts.add(match.group())
            results.append(
                (match.start(), match.end(), self._generate_hash(match.group()))
            )
        return results

    def learn_from_text(self, text: str, known_entities: Dict[str, str] = None):
        """テキストからパターン学習"""
        doc = self.nlp(text)

        # 1. SpaCy エンティティ学習
        for ent in doc.ents:
            if ent.label_ in self.target_entities:
                self.entity_texts.add(ent.text)
                pattern = EntityPattern(
                    text=ent.text,
                    label=ent.label_,
                    pos=ent.root.pos_,
                    features=self._extract_features(ent.root),
                    context=self._get_context(doc, ent.start, ent.end),
                )
                self.learned_patterns[ent.label_].append(pattern)

        # 2. 正規式ベースのエンティティ学習
        for label, pattern in self.entity_patterns.items():
            if isinstance(pattern, list):
                for p in pattern:
                    for match in re.finditer(p, text):
                        self.entity_texts.add(match.group())
            else:
                for match in re.finditer(pattern, text):
                    self.entity_texts.add(match.group())

        # 3. 技術スタック学習
        for pattern in self.tech_patterns:
            for match in re.finditer(pattern, text):
                self.entity_texts.add(match.group())

        # 4. 部署名学習
        dept_matches = self._find_department(text)
        for _, _, hash_val in dept_matches:
            if original := self.reverse_cache.get(hash_val):
                self.entity_texts.add(original)

        # 5. プロジェクト名学習
        project_matches = self._find_projects(text)
        for _, _, hash_val in project_matches:
            if original := self.reverse_cache.get(hash_val):
                self.entity_texts.add(original)

        # 6. ユーザー提供エンティティ学習
        if known_entities:
            for text, label in known_entities.items():
                for match in re.finditer(re.escape(text), doc.text):
                    start, end = match.span()
                    span = doc.char_span(start, end)
                    if span:
                        self.entity_texts.add(text)
                        pattern = EntityPattern(
                            text=text,
                            label=label,
                            pos=span.root.pos_,
                            features=self._extract_features(span.root),
                            context=self._get_context(doc, span.start, span.end),
                        )
                        self.learned_patterns[label].append(pattern)

    def anonymize_text(self, text: str) -> str:
        """テキスト内の敏感な情報をハッシュ化"""
        doc = self.nlp(text)
        replacements = []
        processed_spans = set()

        # 1. 識別子(ID、ライセンス番号)処理
        for pattern in self.entity_patterns.get("ID", []) + self.entity_patterns.get(
            "LICENSE", []
        ):
            id_matches = self._process_identifier(text, pattern)
            for start, end, hash_val in id_matches:
                if self._is_span_available(processed_spans, start, end):
                    processed_spans.add((start, end))
                    replacements.append((start, end, hash_val))

        # 2. 正規式基盤エンティティ処理
        for label, pattern in self.entity_patterns.items():
            if label in ("ID", "LICENSE"):
                continue  # 既に処理済み
            elif label == "MONEY":
                money_matches = self._process_money(text)
                for start, end, hash_val in money_matches:
                    if self._is_span_available(processed_spans, start, end):
                        processed_spans.add((start, end))
                        replacements.append((start, end, hash_val))
        # 3. 技術スタック処理
        tech_matches = self._find_tech_stack(text)
        for start, end, hash_val in tech_matches:
            if self._is_span_available(processed_spans, start, end):
                processed_spans.add((start, end))
                replacements.append((start, end, hash_val))

        # 4. 部署名の処理
        dept_matches = self._find_department(text)
        for start, end, hash_val in dept_matches:
            if self._is_span_available(processed_spans, start, end):
                processed_spans.add((start, end))
                replacements.append((start, end, hash_val))

        # 5. プロジェクト名処理
        project_matches = self._find_projects(text)
        for start, end, hash_val in project_matches:
            if self._is_span_available(processed_spans, start, end):
                processed_spans.add((start, end))
                replacements.append((start, end, hash_val))

        # 6. SpaCy エンティティ処理
        for ent in doc.ents:
            if ent.label_ in self.target_entities:
                start, end = ent.start_char, ent.end_char
                if self._is_span_available(processed_spans, start, end):
                    text_to_hash = ent.text
                    has_title, base_text, suffix = self._has_title_suffix(text_to_hash)

                    if has_title:
                        text_to_hash = base_text

                    # 住所の場合、構成要素別に分離
                    if ent.label_ in ("GPE", "LOC") and any(
                        unit in text_to_hash for unit in self.address_units
                    ):
                        for addr_start, addr_end, hash_val in self._split_address(
                            text_to_hash
                        ):
                            if self._is_span_available(
                                processed_spans, start + addr_start, start + addr_end
                            ):
                                processed_spans.add(
                                    (start + addr_start, start + addr_end)
                                )
                                replacements.append(
                                    (start + addr_start, start + addr_end, hash_val)
                                )
                    else:
                        processed_spans.add((start, end))
                        hashed = self._generate_hash(text_to_hash)
                        if has_title:
                            hashed = f"{hashed}{suffix}"
                        replacements.append((start, end, hashed))

        # 7. 学習されたパターン処理
        for chunk in doc.noun_chunks:
            if chunk.text in self.entity_texts:
                start, end = chunk.start_char, chunk.end_char
                if self._is_span_available(processed_spans, start, end):
                    text_to_hash = chunk.text
                    has_title, base_text, suffix = self._has_title_suffix(text_to_hash)

                    if has_title:
                        text_to_hash = base_text

                    processed_spans.add((start, end))
                    hashed = self._generate_hash(text_to_hash)
                    if has_title:
                        hashed = f"{hashed}{suffix}"
                    replacements.append((start, end, hashed))

        # 8. 最終テキスト変換
        non_overlapping = self._get_non_overlapping_replacements(replacements)
        result = text
        for start, end, replacement in non_overlapping:
            result = result[:start] + replacement + result[end:]

        return result

    def decode_text(self, anonymized_text: str) -> str:
        """ハッシュ化されたテキストを元のテキストに復元"""
        result = anonymized_text
        hash_pattern = r"<<[0-9a-f]{8}>>"

        for match in reversed(list(re.finditer(hash_pattern, result))):
            hash_value = match.group()
            if original_text := self.reverse_cache.get(hash_value):
                start, end = match.span()
                result = result[:start] + original_text + result[end:]

        return result

    def get_statistics(self) -> Dict[str, any]:
        """システム統計情報の返却"""
        entity_counts = defaultdict(int)
        for pattern in self.learned_patterns.values():
            for p in pattern:
                entity_counts[p.label] += 1

        return {
            "total_entities": len(self.hash_cache),
            "unique_hashes": len(set(self.hash_cache.values())),
            "learned_patterns": dict(entity_counts),
            "learned_entities": len(self.entity_texts),
            "entity_types": list(self.target_entities),
            "cache_size": {
                "hash_cache": len(self.hash_cache),
                "reverse_cache": len(self.reverse_cache),
                "entity_texts": len(self.entity_texts),
            },
        }

    def get_entity_mapping(self) -> Dict[str, List[str]]:
        """エンティティタイプ別マッピング情報の返還"""
        mapping = defaultdict(list)
        for pattern in self.learned_patterns.values():
            for p in pattern:
                mapping[p.label].append(p.text)
        return dict(mapping)

    def clear_cache(self):
        """キャッシュ初期化"""
        self.hash_cache.clear()
        self.reverse_cache.clear()
        self.entity_texts.clear()
        self.learned_patterns.clear()
        self.processed_cache.clear()


def main():
    # 1. テストデータ準備
    test_cases = [
        # 基本情報テスト(会社名、人名、連絡先)
        """
        株式会社テスト技研の山田太郎社長(社員番号: EMP123456)は、
        東京都渋谷区にある日本電機株式会社との業務提携を発表しました。
        連絡先：test@example.com、電話：03-1234-5678
        """,
        # 技術スタックテスト
        """
        当社では、PythonとTypeScriptを使用したマイクロサービス開発を行っており、
        インフラにはAWSとKubernetesを採用。データベースはMySQLとMongoDBを使用しています。
        また、新規プロジェクトではReactとDjangoの組み合わせを検討中です。
        """,
        # 部署/プロジェクト情報テスト
        """
        技術開発部第一課のプロジェクトリーダーである佐藤部長は、
        研究開発部及びIT推進部と協力し、新システム「Project-X」の開発を進めています。
        進捗状況は毎週金曜日にSlackで共有されています。
        """,
        # 金額/日付情報テスト
        """
        2024年3月期の売上高は¥1,234,567,890円で、
        前年比25.5%増となりました。新規プロジェクトは2024/4/1 10:30から
        開始予定です。予算は約￥50,000,000円を見込んでいます。
        """,
        # 資格・免許情報テスト
        """
        情報処理安全確保支援士（登録番号：第12345号）の資格を持つ
        田中氏が、セキュリティ監査を担当します。
        詳細は https://example.com/audit/report を参照してください。
        """,
        # 複合情報テスト
        """
        株式会社テクノソリューションズ（本社：東京都千代田区丸の内1-1-1）は、
        クラウドサービス「CloudHub-X」（特許第123456号）の提供を開始しました。
        本サービスは、AWSとGCP両環境に対応し、Kubernetes上で動作します。
        開発責任者の鈴木CTO（技術開発本部長）は、「約￥10億円を投資し、
        第一開発部と品質管理部の総勢50名体制で開発を進めてきました」と説明。
        問い合わせ先：info@example.com（クラウドサービス推進室）
        """,
    ]

    # 2. 既知のエンティティ定義
    known_entities = {
        # 製品/サービス
        "Amazon EC2": "PRODUCT",
        "Google Cloud": "PRODUCT",
        "AWS Lambda": "PRODUCT",
        "Docker": "PRODUCT",
        "Kubernetes": "PRODUCT",
        "Slack": "PRODUCT",
        "CloudHub-X": "PRODUCT",
        # プロジェクト
        "Project-X": "PROJECT",
        "Phoenix": "PROJECT",
        "Symphony": "PROJECT",
        # システム
        "SystemOne": "SYSTEM",
        "CloudBase": "SYSTEM",
        "DataHub": "SYSTEM",
    }

    # 3. テスト実行
    anonymizer = JapaneseTextAnonymizer()

    print("=== 初期学習中... ===")
    # 1つ目と2つ目のケースで初期学習
    for text in test_cases[:2]:
        anonymizer.learn_from_text(text)
    # 既知のエンティティで追加学習
    anonymizer.learn_from_text(test_cases[2], known_entities)

    # 各ケーステスト
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n=== テストケース {i} ===")
        print("原本テキスト:", test_text)

        # 익명화
        anonymized = anonymizer.anonymize_text(test_text)
        print("\n匿名化されたテキスト:", anonymized)

        # 복원
        decoded = anonymizer.decode_text(anonymized)
        print("\n復元されたテキスト:", decoded)

        # 검증
        is_identical = decoded.strip() == test_text.strip()
        print("\n復元成功:", "Yes" if is_identical else "No")

        if not is_identical:
            print("\n相違点:")
            print("原本:", repr(test_text.strip()))
            print("復元.:", repr(decoded.strip()))

    # 4. システムの統計
    print("\n=== システムの統計 ===")
    stats = anonymizer.get_statistics()

    print("\n処理されたエンティティ:")
    for key, value in stats["learned_patterns"].items():
        if value > 0:
            print(f"- {key}: {value}個")

    print("\n総計:")
    print(f"- 全エンティティ数: {stats['total_entities']}")
    print(f"- 固有ハッシュ数: {stats['unique_hashes']}")
    print(f"- 学習されたエンティティ: {stats['learned_entities']}")
    print("\nキャッシュのサイズ:")
    for key, value in stats["cache_size"].items():
        print(f"- {key}: {value}")

    # ユーザー入力テキスト
    user_text = """ここにユーザーの入力が入ります"""

    #  ユーザーテキスト処理
    print("\n=== ユーザーテキスト処理 ===")
    print("原本テキスト:", user_text)

    # 익명화
    anonymized = anonymizer.anonymize_text(user_text)
    print("\n匿名化されたテキスト:", anonymized)

    # 복원
    decoded = anonymizer.decode_text(anonymized)
    print("\n復元されたテキスト:", decoded)

    # 검증
    is_identical = decoded.strip() == user_text.strip()
    print("\n復元成功:", "Yes" if is_identical else "No")

    if not is_identical:
        print("\n相違点:")
        print("原本:", repr(user_text.strip()))
        print("復元:", repr(decoded.strip()))

    # 4. 処理されたエンティティ情報の出力
    print("\n=== システムの統計 ===")
    stats = anonymizer.get_statistics()

    print("\n処理されたエンティティ:")
    for key, value in stats["learned_patterns"].items():
        if value > 0:
            print(f"- {key}: {value}個")

    print("\n総計:")
    print(f"- 全エンティティ数: {stats['total_entities']}")
    print(f"- 固有ハッシュ数: {stats['unique_hashes']}")
    print(f"- 学習されたエンティティ: {stats['learned_entities']}")


if __name__ == "__main__":
    main()
