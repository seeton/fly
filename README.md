# fly — ハエの脳(コネクトーム)を触る環境

ショウジョウバエの神経系を電子顕微鏡で丸ごと輪切りにして、全ニューロンと全シナプスを
再構築した「コネクトーム」を手元で触るための環境。

## 入っているデータ

| | **Male CNS v1.0** | **hemibrain** | **FAFB-FlyWire** |
|---|---|---|---|
| 範囲 | オスの**中枢神経系まるごと**<br>(脳＋視葉＋腹部神経索) | メス脳の中心部のみ<br>(視葉なし) | メスの脳まるごと<br>(神経索なし) |
| ボディ数 | 211,577 | 21,739 | 139,248 |
| 接続 | **1億5185万** (3億1183万シナプス) | 355万 | — (注釈のみ) |
| 出典 | Janelia FlyEM 2025 | Scheffer et al. 2020 | Dorkenwald/Schlegel et al. 2024 |
| ログイン | 不要 | 不要 | 注釈は不要 / 生データは要アカウント |

**Male CNS が主力**。脳と神経索が繋がっているので「見る → 判断する → 脚を動かす」を
1本の配線図の上で端から端まで辿れる。hemibrain は軽くてツールが全部対応しているので練習用、
FlyWire はオスとメスを比べるための対照。

## セットアップ済みのもの

- Python 3.12 (uv 管理) + 仮想環境 `.venv/`
- `navis` (形態解析・3D可視化), `fafbseg` (FlyWire), `neuprint-python`, `flybrains`,
  pandas / pyarrow / networkx / plotly / JupyterLab
- `flygym` + `mujoco` (NeuroMechFly v2 / flybody = 物理エンジン上のハエの体)
- `fastapi` + `uvicorn` (ローカルアプリ)

## 使い方

```powershell
cd C:\Users\seeton\fly
.\.venv\Scripts\Activate.ps1
```

(有効化せずに `.\.venv\Scripts\python.exe` を直接叩いてもよい)

### まず 3D で見る

```powershell
python scripts\02_neuron_3d.py
```

hemibrain の実物のニューロン(嗅覚投射ニューロン DA1 uPN)5本と脳メッシュがブラウザで開いて回せる。

### 逃避回路を辿る — Male CNS

```powershell
python scripts\05_explore_malecns.py --descending DNp01
```

ハエに手を伸ばすと飛んで逃げる、あの回路。出力:

```
    10001  DNp01  [descending_neuron]
 -> 800146  TTMn   [vnc_motor]
```

DNp01 = Giant Fiber(脳で一番太い軸索を持ち、1本で首を貫いて神経索まで降りる)。
TTMn = 中脚を蹴り出して体を空中に打ち上げる筋肉の運動ニューロン。**単シナプス**で直結
＝最速で逃げるための配線で、数十年の電気生理の結論がそのままデータから出てくる。

`--type DNp01` で上流を見ると1位・2位が **LC4 / LPLC2** = 迫ってくる影の検出器。
つまり 視覚 → 巨大繊維 → 脚の筋肉 が一続きで見える。

```powershell
python scripts\05_explore_malecns.py                  # 全体サマリ
python scripts\05_explore_malecns.py --search pC1     # 細胞型を検索
python scripts\05_explore_malecns.py --type DNp01     # 上流/下流
python scripts\05_explore_malecns.py --dimorphic      # 性的二型のニューロン
```

### hemibrain / FlyWire

```powershell
python scripts\01_explore_hemibrain.py --type MBON01   # 学習出力ニューロンの上流/下流
python scripts\03_flywire_annotations.py --nt          # メス脳の神経伝達物質の内訳
```

### ノートブック

```powershell
jupyter lab
```

- `notebooks\01_first_touch.ipynb` — hemibrain。3D表示 → 配線表 → MBONの上流 → 最短経路。
  「フェロモン入力 DA1_lPN から学習出力 MBON01 まで」を引くと `DA1_lPN → KCg-m → MBON01` と
  教科書通りキノコ体を経由する経路が出る
- `notebooks\02_malecns.ipynb` — Male CNS。逃避回路、求愛回路(pC1)、性的二型、
  FlyWire(メス)との細胞数比較。**実行済みの出力付き**

---

## ビジュアル (out/ に出力)

| スクリプト | 出力 | 中身 |
|---|---|---|
| `02_neuron_3d.py` | neurons_3d.html | hemibrain のニューロン + 脳メッシュ |
| `06_escape_circuit_3d.py` | escape_circuit_3d.html | 逃避回路の骨格。青=LC4/水色=LPLC2 → 赤=DNp01 → 緑=TTMn |
| `07_synapses_3d.py` | synapses_3d.html | hemibrain の実測シナプス点群 (水色=入力/赤=出力) |
| `08_malecns_synapses_3d.py` | malecns_synapses_3d.html | MaleCNS の実測シナプス。`--partner TTMn` で接触点を強調 |
| `09_escape_animation.py` | escape_animation.html | **▶再生**: 実際の軸索に沿って信号が脳→神経索へ降りる |
| `10_fly_walk.py` | fly_walk.mp4 | 物理エンジンの中でハエが歩く (三脚歩行) |
| `11_escape_takeoff.py` | escape_takeoff.mp4 | **回路が指した動作を体にやらせる**: DNp01発火→中脚伸展→跳躍 |
| `12_flight.py` | flight.mp4 | **飛ぶ**: 200Hzの羽ばたきで体重の1.16倍の揚力 |

```powershell
python scripts\09_escape_animation.py                     # 信号が降りるアニメ
python scripts\08_malecns_synapses_3d.py --partner TTMn   # GF→TTMn の90シナプス
python scripts\11_escape_takeoff.py                       # 跳躍 (体長の73%浮く)
python scripts\10_fly_walk.py --seconds 2                 # 歩行
```

### 回路から動作まで一続きで分かること

`--partner TTMn` で出る **90シナプス**（全部 LTct 内）が Giant Fiber と中脚運動
ニューロンの接触点。TTM は中脚の筋肉なので、`11_escape_takeoff.py` で中脚だけを
蹴ると体長の 73% 浮くが、`--with-hind` で後脚も一緒に蹴ると **かえって低くなる**
(+1.83 → +1.67 mm)。解剖の通りになる。

## ハエの体 (NeuroMechFly v2)

`flygym` = マイクロCTから作られた実物大のハエの3Dモデルを MuJoCo で動かすもの。
6本脚・66関節。初回実行時にメッシュ (約200MB) を自動ダウンロードする。

**つまづきやすい点**: 位置アクチュエータの `kp` を上げないと (既定値では)
関節が指令に全く追従せず、その場で震えるだけになる。ここでは `kp=45` を使っている。

いま世界でやられているのは、この体に**コネクトームをそのまま繋ぐ**こと:

- **NeuroMechFly v2 / flygym** — 体の側。ここに入れている
  https://neuromechfly.org
- **Flyvis** (TuragaLab) — FlyWire の視覚系の結線でネットワークを作り、
  実際に記録された神経活動を予測できた (Nature 2024)
  https://github.com/TuragaLab/flyvis
- **全脳埋め込みシミュレーション** — FlyWire 13.8万ニューロンのスパイキングモデルを
  NeuroMechFly の体に入れて、見る・歩く・毛づくろい・逃げるを出させる試み
  https://github.com/cobanov/awesome-fly (まとめ)

ここにある `11_escape_takeoff.py` は、その最小版
(回路の追跡結果を手で体の指令に翻訳したもの) にあたる。

---

## アプリ — ハエ脳ブラウザ

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.server:app --port 8765
```

→ http://localhost:8765 (起動時に1GBを読むので20秒ほどかかる)

左で細胞型を検索 → 選ぶと上流/下流がシナプス数順に出る。相手をクリックすれば
そのまま辿っていける。右は 3D ビュー。

| ボタン | すること |
|---|---|
| 骨格 | 選んだ細胞型の 3D 形状 |
| シナプス | 6.8GB を走査して実測シナプスを点群表示 (2〜3秒) |
| 接触点 | 「接触相手」に入れた細胞型との接触シナプスだけ黄色で表示 |
| 運動ニューロンまで何段？ | weight≥10 の接続で何ステップで筋肉に届くか |
| 経路探索 | 2つの細胞型の間の最短経路を求めて 3D に色分け表示 |
| 動画 | 飛行・跳躍・歩行のシミュレーション |

試すと面白い例:

- `DNp01` を選び、接触相手に `TTMn` → 接触90シナプスが全部 LTct にあるのが見える
- `DNp01` で「運動ニューロンまで何段？」→ 815個中774個に4段以内で届く。
  1段が TTMn (跳躍)、2段に **DLMn / DVMn = 飛翔筋の運動ニューロン**
- `LC4` → `DNp01` → `TTMn` と辿ると、目から筋肉までが一本道でつながる

## 飛ばす

```powershell
python scripts
_flight.py                # 飛ぶ
python scripts
_flight.py --aero plain   # 比較: 落ちる
```

Janelia flybody (翅つき全身モデル) を 200 Hz で羽ばたかせる。翅の付け根の3自由度は

- **roll** = ストローク (前→後ろに扇ぐ) ← 主動力
- **pitch** = 迎角 (打ち返しで反転)
- **yaw** = 展開 (畳む↔広げる)

結果: 高度 1.00 → 1.63 cm、上昇速度 39.6 cm/s、正味空力は体重の 1.16 倍。

### そのままでは飛ばない — 埋めた穴が2つ

1. **メッシュ名の不一致**: `fruitfly.xml` は `wing_left_brown.obj` を参照するが、
   配布メッシュは `l_wing_brown.obj`。rigging.yaml の `flybody_name` から
   対応表を作って解決した (85個すべて対応)。[scripts/flybody_model.py](scripts/flybody_model.py)
2. **空力が無効**: 翅に楕円体の「流体ジオム」はあるのに `fluidshape="ellipsoid"`
   が設定されておらず、羽ばたいても力が出ない。`geom_fluid` を立てて有効化した。

さらに翅アクチュエータは既定の設定では指令にまったく追従しない
(翅の armature が実効慣性を支配している)。位置サーボ化し `kp=50`、
タイムステップ 2e-5 で追従するようにした。

### 空力係数について (重要)

MuJoCo の楕円体流体モデルは、昆虫が揚力を稼ぐ **前縁渦 (LEV)** を表現しない。

- `--aero plain` (MuJoCo 既定): 体重の **0.79 倍** → 落ちる
- `--aero real` (Kutta揚力係数 3, blunt抗力 1.5): 体重の **1.43 倍** → 飛ぶ

後者は「昆虫の実測揚力係数は定常翼理論の2〜3倍」という事実に合わせた
粗い代表であって、前縁渦を解いているわけではない。両方入れてあるので比較できる。

羽ばたきパラメータ (周波数・ストローク振幅・迎角・位相の8個) は
正味空力の大きさを目的関数にしたランダム探索で決めた。見つかった最適値は
**200 Hz・ストローク振幅ほぼ最大** で、実際のショウジョウバエとほぼ同じだった。

## 飛び方を学習させる (報酬の的を少しずつ前に置くカリキュラム)

```powershell
python scripts\13_train_flight.py       # 学習 (30コア並列, 約30分)
python scripts\14_render_flight.py      # 学習した飛び方を録画
python scripts\14_render_flight.py --no-control   # 制御を切ると宙返りする
```

手でゲインを合わせるかわりに CMA-ES で制御器16パラメータを探索する。
制御器は本物のハエの平均棍(haltere)反射と同じ構造:
ピッチ←ストローク中心、ロール←左右の振幅差、ヨー←左右の迎角差、
高度←振幅の増減。的を「その場→1cm前→3cm前→8cm前」と段階的に遠ざける。

### 到達した状態 (1.5秒飛行)

| 項目 | 値 |
|---|---|
| 高度保持 | 9.83 cm ± **0.03 cm** |
| ロールの振れ | **±0°** |
| ヨーの総回転 | **0°** (1.5秒) |
| 横方向のずれ | **-0.1 cm** (ほぼ直線) |
| 前進 | 24 cm / 1.5 s ≒ 16 cm/s |
| 羽ばたき | **210 Hz** (実物は約200 Hz) |

### まだ直っていない: 機首下げ約50度

本物のショウジョウバエは低速飛行で**機首上げ30〜45度**。この方策は
**機首下げ50度**で飛ぶ。ピッチ目標を振っても到達できる範囲は -32〜-75度で、
機首上げ側には入れない (それ以上振ると墜落する)。

原因はおそらく制御器の構造にある。いまは翅の展開角 (yaw) を**一定に固定**して
ストローク(roll)と迎角(pitch)だけを振っているので、翅先は単純な往復軌道を描く。
本物のハエは展開角も羽ばたき周期内で振って**8の字軌道**を描き、それで
ストローク面の向きを体の姿勢と独立に選べる。次にやるならそこ。

### 報酬設計で踏んだ失敗 (全部ログに残してある)

| 書き忘れたもの | 起きたこと |
|---|---|
| 目標高度 | 姿勢は完璧(傾き2°)だが揚力を捨てて落下。係留揚力 1.08→0.44 |
| 減点を有界にする | 「早く死ぬほど高得点」になり、飛べる解(報酬-4.4)を捨てて即墜落(+0.003)を選んだ |
| 機首方向 | 水平は保つがコマのように回転 (1秒で-242度) |
| 傾きの**向き** | 機首下げ50度で安定してしまう |

「報酬に書かなかったものは最適化されない」という同じ失敗を4回繰り返した。


## 体のモデルは正しいのか — 検証の記録

「飛ばない/不自然な飛び方をする」原因を、報酬をいじる前に体の側で突き止めた記録。

### 正しかったもの

| 検査項目 | 結果 |
|---|---|
| 全身質量 | 0.985 mg (実物 約1.0 mg) |
| メッシュ85件の割り当て | 語幹不一致 0件 (貪欲マッチは壊していなかった) |
| 重心の位置 | 翅の付け根の 281 um 後方・254 um 下 — 解剖学的に妥当 |
| 翅の長さ | 約 2.3 mm (実物どおり) |

### 間違っていたもの 1 — ストローク面が固定されていた

翅の付け根は3自由度 (ストローク・迎角・**展開角**) あるのに、展開角を
一定に固定してストロークと迎角だけ振っていた。すると翅先が単純な往復軌道
しか描かず、掃く面の向きが体に固定される。

機体を水平に係留して正味の空力を測ると **体軸の真上から34度ずれて後ろ向き**。
この力を鉛直にするには機体を傾けるしかない = 機首下げ50度の正体。

展開角も周期内で振ると向きを選べるようになる:

| 展開角の振幅 | 体軸zからの力の傾き |
|---|---|
| 0.0 (従来) | -34度 |
| 0.8 | -12度 |
| 1.2 | **+1度** (真上) |

### 間違っていたもの 2 — 空力係数を非対称に水増ししていた

Kutta揚力だけ3倍・blunt抗力だけ3倍という改変をしていた。これは力の
大きさだけでなく **向きまで歪める**。一様スケールに改めた (既定の2倍)。
掃く面を水平にすれば一様2倍でも最大 **2.35 体重ぶん** 出るので、
非対称な3倍は不要だった。

### 間違っていたもの 3 — 翅の慣性が armature で水増しされていた

元の XML は翅の関節に `armature=1e-6` を入れているが、これは翅自身の
ヒンジまわりの慣性 `2.3e-7` の **4.3倍**。数値安定化の項が実際の慣性を
上回っていた。実慣性に合わせると羽ばたきのパワーが半減する
(32,000 -> 14,700 erg/s)。`load_model(wing_armature=...)` で既定修正済み。
ただし 5e-8 まで下げると dt=1e-5 でも発散する。

### 残る本質的な限界 — 空力モデルが昆虫飛行のものではない

消費パワーを係留状態で実測した:

| 条件 | パワー | 揚力 |
|---|---|---|
| 学習後の羽ばたき (260Hz, 振幅1.25) | 14,700 erg/s | 1.04 体重 |
| 実物寄り (200Hz, 振幅1.0) | 3,200 erg/s | 0.58 体重 |
| **実物のショウジョウバエ** | **300-800 erg/s** | **1.0 体重** |

**実物の約4倍のパワーを使って6割の揚力しか出ない** = 効率が約7倍悪い。

MuJoCo の楕円体流体モデルは、昆虫が揚力を稼ぐ **前縁渦 (LEV)** や
**回転揚力** を表現しない。だから係数を水増ししないと浮かないし、
水増ししても力の **向き** は正しくならない。機首下げ姿勢はその帰結で、
報酬をどういじっても直らない (実際、体軸角ボーナスを入れても -57度のまま
動かなかった)。

**次にやるなら**: MuJoCo の汎用流体モデルを使うのをやめ、
Sane & Dickinson の準定常翼素理論 (並進・回転・付加質量の3項) を
自分で実装して、毎ステップ翅に外力として加える。体のモデルはそのまま使える。


## 空力を昆虫のものに入れ替えた

MuJoCo の楕円体流体モデルをやめ、**準定常翼素理論** (Sane & Dickinson 2002) を
実装した。翅をスパン方向に分割し、翼素ごとに3項を毎ステップ計算して外力で与える。

```powershell
python scripts\16_validate_aero.py            # 検証
python scripts\16_validate_aero.py --sweep    # 迎角と振幅を振る
```

- **並進力** — 力係数は Dickinson et al. (1999) がショウジョウバエの翅の
  力学的スケールモデルで実測した値。前縁渦 (LEV) の効果込みで
  `C_L` は迎角45度付近で **1.8** に達する (定常翼理論なら約0.9)。
- **回転揚力** (Kramer効果) — 打ち返しで翅がひねられるときに出る力。
- **付加質量** — 翅が押しのける空気の慣性。

### 都合の悪い物理も残す

翅だけを内蔵流体から外し、**胴体・脚・頭・腹部の空気抵抗は MuJoCo に残している**。
翅の流体ジオムを「楕円体モデル・係数ゼロ」にすると、そのジオムには力が出ず、
その body は内蔵の inertia-box モデルからも外れる。`opt.density` はそのままなので
胴体の抗力は生きる (実測: 100 cm/s 前進時の減速度は処理の前後で
-505.7 → -502.8 cm/s^2 とほぼ不変)。

翅にも抗力はかかる。迎角90度では `C_D = 3.46` で、揚力より大きい。

### XML の数値安定化パラメータが物理を壊していた

`fruitfly.xml` の翅関節の値は、この寸法にはどれも大きすぎた。

| 項目 | XML の値 | 実際の物理 | 影響 |
|---|---|---|---|
| `armature` | 1e-6 | 翅の慣性は 2.3e-7 | 実効慣性が4.3倍 → パワー倍増 |
| `damping` | 5e-4 | ほぼ無視できるはず | 1400 rad/s で 0.7 dyn cm の抵抗 → **パワーの大半をここで捨てていた** |

ダンピングを 5e-4 → 1e-5 にすると正味パワーが **2381 → 683 erg/s**。
揚力は変わらない。`load_model(wing_armature=..., wing_damping=...)` で既定修正済み。

### パワーの測り方

`sum |F v|` (絶対値) で測ると共振で戻ってくるぶんまで費用に数えてしまう。
本物の昆虫は胸部がバネとして働き慣性エネルギーを回収するので、
**正味の `sum F v`** で測るのが正しい。絶対値だと 9080、正味だと 2381 erg/s と
4倍近く違った。

### 検証結果

240 Hz・ストローク振幅 1.25 rad・迎角 50 度・機首上げ 50 度:

| | MuJoCo 内蔵の流体 | **翼素理論** | 実物のショウジョウバエ |
|---|---|---|---|
| 揚力 | 0.58 体重 | **0.951 体重** | 1.0 |
| 正味パワー | 3,200〜14,700 erg/s | **1,001 erg/s** | 300〜800 |
| 釣り合う姿勢 | 機首**下げ** 50度 | **機首上げ 50度** | 機首上げ 30〜45度 |
| 翅先の速度 | — | 230〜280 cm/s | 200〜300 |
| 迎角 | — | 46〜50度 | 約45度 |

**姿勢の符号が反転した**のが最大の変化。内蔵流体では正味の力が体軸から
後ろ向きにずれていて、機体を機首下げにしないと鉛直にならなかった。
翼素理論では前向きにずれるので、実物と同じ**機首上げ**姿勢で釣り合う。
報酬をいじっても直らなかったのは当然で、空力モデルの問題だった。

揚力 0.951 はホバリングまであと一歩 (羽ばたきの最適化余地)。
パワーはまだ実物の1.3〜3倍。


---

## ディレクトリ

```
fly/
├─ .venv/
├─ data/
│  ├─ malecns/                                 Male CNS v1.0 (1.1 GB)
│  │    body-annotations-...feather              21万ボディの注釈(型/左右/性的二型/他DS対応)
│  │    connectome-weights-...feather            全結合 1億5185万本
│  │    syn-partners-...feather                  全シナプスの3D座標 (6.8 GB)
│  │    body-neurotransmitters-...feather        神経伝達物質の推定
│  │    skeletons/                               個別ニューロンの骨格 (必要時に取得)
│  ├─ hemibrain/exported-traced-adjacencies-v1.2/
│  │    traced-neurons.csv / traced-total-connections.csv / traced-roi-connections.csv
│  └─ flywire/flywire_783_annotations.tsv       13.9万ニューロンの注釈
├─ notebooks/  01_first_touch.ipynb  02_malecns.ipynb
├─ scripts/
│  ├─ 00_fetch_data.py          データ取得 (--malecns で Male CNS も)
│  ├─ 01_explore_hemibrain.py
│  ├─ 02_neuron_3d.py
│  ├─ 03_flywire_annotations.py
│  ├─ 04_flywire_live.py        FlyWire サーバから生データ (要トークン)
│  ├─ 05_explore_malecns.py
│  ├─ 06_escape_circuit_3d.py   逃避回路の3D
│  ├─ 07_synapses_3d.py         hemibrain のシナプス点群
│  ├─ 08_malecns_synapses_3d.py MaleCNS のシナプス点群
│  ├─ 09_escape_animation.py    信号伝播アニメ
│  ├─ 10_fly_walk.py            歩行シミュレーション
│  ├─ 11_escape_takeoff.py      跳躍シミュレーション
│  ├─ 12_flight.py              飛行シミュレーション
│  ├─ flybody_model.py          翅つきモデルの読み込み(メッシュ名解決+空力有効化)
│  ├─ flight_env.py             飛行の評価環境と報酬
│  ├─ 13_train_flight.py        CMA-ES + カリキュラム学習
│  ├─ 14_render_flight.py       学習した飛び方を録画
│  ├─ 15_bootstrap_flight.py    学習の出発点をランダム探索+CMA-ESで探す
│  ├─ insect_aero.py           準定常翼素理論の空力 (Sane & Dickinson)
│  └─ 16_validate_aero.py      空力の検証 (揚力・パワー・姿勢)
├─ app/
│  ├─ server.py                 FastAPI バックエンド
│  └─ static/index.html         フロントエンド
├─ out/                         生成した HTML
└─ requirements.txt
```

## Male CNS の注釈で使える主な列

`superclass` (descending_neuron / vnc_motor / ol_intrinsic など), `type`, `instance`,
`somaSide`, `somaNeuromere` (T1/T2/T3 = 前脚/中脚/後脚の節), `entryNerve` / `exitNerve`,
`dimorphism`, `fruDsx`, そして **`flywireType` / `hemibrainType` / `mancType`**
(他データセットとの対応付け。FlyWire とは 143,156 件が対応)。

内訳: 下行ニューロン 1,314 / 上行 1,846 / 運動 815 / オス特異的 1,258。

## もっと欲しくなったら

- **ブラウザで3D**: https://male-cns.janelia.org (bodyId 検索) / https://codex.flywire.ai / https://neuprint.janelia.org
- **FlyWire の生の接続データ**: 無料登録 → https://join.flywire.ai/ →
  トークン https://global.daf-apis.com/auth/api/v1/user/token →
  `python scripts\04_flywire_live.py --token <トークン>`
- **骨格(3D形状)**: Male CNS の SWC は `gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/`
- **BANC** (メスの脳＋神経索, Nature 2026): https://doi.org/10.7910/DVN/7WTH1N
  — 入れればオス(Male CNS)とメス(BANC)の全CNS比較ができる

## データの出典 (論文で使うときは要引用)

- Male CNS v1.0 — Janelia FlyEM / Google Research (CC BY 4.0)
  https://www.janelia.org/project-team/flyem/male-cns-connectome
- hemibrain — Scheffer et al. (2020) *eLife* (CC BY 4.0)
- FlyWire 注釈 — Schlegel et al. / Dorkenwald et al. (2024) *Nature* (CC BY 4.0)

## 環境を作り直す場合

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
python scripts\00_fetch_data.py --all
```
