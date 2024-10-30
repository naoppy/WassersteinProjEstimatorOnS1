# About
フォンミーゼス分布や巻き込みコーシー分布などの超球上の分布に対してMLEとW-estimatorの比較をする

## von Mises Distribution
平均パラメータ `mu` と尖度パラメータ `kappa` をもつ。
確率密度関数は
```
f(theta) = exp{kappa cos(theta - mu)} / (2pi I_0(kappa))
```
ここで、I_j は j次の第一種変形ベッセル関数である。

フォンミーゼス分布は指数型分布族に属するので、その理論を応用できる。
十分統計量は `sin theta` と `cos theta` (の和) である。

### 最尤推定
`mu` と `kappa` のどちらも未知のとき、最尤推定量は
```
mu = tan^{-1} (sum_i (sin(theta_i)) / sum_i (cos(theta_i)))
```
```
A(kappa)
:= I_1(kappa) / I_0(kappa)
 = 1/N(sum_i cos(theta_i)) cos(mu_MLE) + 1/N(sum_i sin(theta_i)) sin(mu_MLE) 
```
となる。kappaについては、上記の式を満たす `kappa` を数値的に求めるほかない。
I_1/I_0 は単調増加な関数なので、二分探索を用いて計算するのがよいだろう。

### W-estimator
