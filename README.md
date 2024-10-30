# About
フォンミーゼス分布や巻き込みコーシー分布などの超球上の分布に対してMLEとW-estimatorの比較をする

## von Mises Distribution
平均パラメータ mu と尖度パラメータ beta をもつ。
確率密度関数は
f(theta) = exp{beta cos(theta - mu)} / (2pi I_0(beta))
ここで、I_j は j次の第一種変形ベッセル関数である。

### 最尤推定
