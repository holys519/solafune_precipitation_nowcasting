# Competition Rules Notes

最終確認: 2026-07-16 UTC

確認元:

- [公式コンペページ](https://community.solafune.com/competitions/f87811b8-1964-4f4b-84b3-6fddd67ec4b1)
- [公式competition API](https://production.server.solafune.com/api/v3/competition/f87811b8-1964-4f4b-84b3-6fddd67ec4b1)
- `discussion/approved_geocoding_sources_ja.md`（公式回答のローカル和訳。ただしpermalink未記録）

公式締切は2026-07-27 23:59 GMT。提出上限は1日あたりチーム人数×5回で、GMT 00:00
（JST 09:00）にresetされる。

## External Data and Pretrained Weights

外部データセットは禁止。Solafune配布データと、そこから作成した特徴・統計・モデル出力を
基本入力とする。外部衛星画像、追加IMERG、PMW/radar、NWP/再解析、DEM、土地被覆、気候値、
外部product由来のLUT・正規化統計等を結合しない。

一方、公式ページが明示する次のライセンスの公開program、pretrained model、weightは条件付きで
利用できる。

- CC (0, 1.0, 2.0, 3.0, 4.0) / CC BY (1.0, 2.0, 3.0, 4.0)
- MIT / BSD / Apache License 2.0
- US Public Domain

有料、排他的、非商用限定、権利・出典不明のassetは使わない。利用する場合は、model名だけで
なく、checkpoint URL、version、license、取得日、SHA-256、load箇所をmanifestへ保存する。

## Real-world Applicability and Leakage

公式規約は、実世界で適用可能な堅牢なモデルを期待し、すでに存在するground truthの利用や
reverse engineeringに依存する手法を避けるよう求めている。このため現在は次のように扱う。

| 手法 | 状態 | 最終方針 |
| --- | --- | --- |
| 当該CSV rowに列挙された衛星画像だけで推論 | green | 利用可 |
| 配布データだけから作るspectral/temporal特徴・補助task・OOF calibration | green | 利用可 |
| 許可済みgeocoderとsolar geometry | amber→green | 公式topic/message証跡を回収後に利用可 |
| successor rowの対象時刻以後の衛星画像 | amber | 運営の書面回答まで最終提出から除外 |
| evaluation複数rowを読むtemporal filter/postprocess | amber | causal/bidirectional、3/5-tapを問わず回答まで除外 |
| evaluation画像によるtransductive SSL/adaptation | amber | 運営の書面回答まで最終提出から除外 |
| 外部仕様によるband mapping、衛星直下点、数値SRF、外部校正式・固定係数 | amber | 特徴へ埋め込む前に運営確認 |
| train targetをevaluation重複画素へcopyするpatch | red | 最終提出に使わない |
| evaluation `test_files` placeholderの値・統計 | red | 読み込まない |

「submission valid」はzip形式の検証であり、手法の規約承認ではない。
blend、calibration、controller、fine-tune等の派生artifactは、全upstreamのうち最も厳しい
risk区分を継承する。

## Geocoding Exception

公式回答で許可されたsourceはGeoNamesとNominatim/OpenStreetMapのみ。EPSG:4326/WGS84の
canonical recordを使い、source、record ID/query URL、canonical name、lat/lon、取得日、
変換scriptを固定保存する。

現状のローカル和訳には公式discussionのpermalink、topic/message ID、投稿日が残っていない。
最終利用前にログイン済み公式ページからこれらを回収し、回答本文の取得日・checksumまたは
screenshotも監査証跡へ保存する。それまでは新しいcoordinate-derived modelをgreen finalへ
昇格させない。

座標から閉形式で作る緯度経度、半球、sin/cos、local solar time、solar geometryは利用可。
標高、海岸線距離、気候区分、過去降水、気温・湿度・風、土地被覆等の外部結合特徴は禁止。

## Metric Ambiguity

公式ページの一般式と公式utilityは全要素pool RMSEと読めるが、ローカルOOFでは
`mean_tiles(sqrt(mean_pixels(error^2)))`がPublic LBを大幅によく説明する。ただし近縁modelを
tile指標で選択してきたselection biasとOOF/Public distribution shiftがある。hidden evaluatorの
aggregationを運営へ確認するまで、どちらかを公式実装と断定しない。監査結果は
`outputs/l_eda/exp003/CV_LB_CALIBRATION.md` と
`doc/research_survey_v3_2026-07-16.md` §4に記録する。

## Submission Format

```text
submission.zip
├── evaluation_target.csv
└── test_files/
    ├── {location}_GPM_IMERG_{datetime}.tif
    └── ...
```

## Reproducibility

上位候補者は、コンペ終了後にソースコード、解法記事URL、本人確認等を求められる。実験ごとに
以下を残す。

- 実験IDとGit差分
- データ前処理
- CV分割
- 学習設定
- 推論設定
- 後処理
- 提出zipの作成手順
- data / weight / outputのhashとlicense manifest

学習toolはopenかつ無料のものに限定する。最終検収コードは少なくとも前処理、学習、予測を
分離し、それぞれの実行でartifactを再生成できる形にする。

コンペ中、配布データやそこから作ったsecondary productを公開しない。最終提出するソースコードは
team外の第三者へ共有しない。一般的な自作algorithm・idea・資料を公開する場合は全参加者が
閲覧できる状態にし、公式discussionの公開手順に従う。特定の他teamへ私的に渡さない。
コンペ終了後は規約に従ってSolafune配布データを直ちに削除する。
