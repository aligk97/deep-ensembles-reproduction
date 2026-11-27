# Temel denklemler ve türetim adımları

Bu not, rapordaki İngilizce notation'ı Türkçe açıklamalarla tekrarlar. Her üye başlangıç varsayımlarından sonuca gidebilmelidir. $x\in\mathbb{R}^d$, minibatch $X\in\mathbb{R}^{B\times d}$, sınıf sayısı $K=10$, ensemble boyutu $M$, deney seed sayısı $S=3$.

## 1. Proper log score ve KL

$$\mathbb E_q[-\log p(Y|x)]
=-\mathbb E_q\log q(Y|x)+\mathbb E_q\log\frac{q(Y|x)}{p(Y|x)}
=H(q)+D_{KL}(q\Vert p).$$

İlk terim p'den bağımsız; KL≥0. Dolayısıyla population optimum p=q'dur. Finite-data calibration veya OOD garantisi çıkmaz.

## 2. Gaussian NLL

$$p(y|x)=\frac{1}{\sqrt{2\pi v}}\exp\left[-\frac{(y-\mu)^2}{2v}\right],\quad v>0.$$

$$-\log p(y|x)=\tfrac12\log(2\pi)+\tfrac12\log v+\frac{(y-\mu)^2}{2v}.$$

Density'nin çarpımını log toplamına, üs terimini residual/variance terimine çevirin. Training'de sabit atılabilir; evaluation'da density constant/original units korunur. Fixed v durumunda objective MSE'ye orantılıdır.

## 3. Softplus ve target normalization

$$v=\operatorname{softplus}(a)+\epsilon_v=\log(1+e^a)+\epsilon_v,\quad \epsilon_v=10^{-6}.$$

$$\frac{d\,\operatorname{softplus}(a)}{da}=\frac{1}{1+e^{-a}}.$$

$$\widetilde y=(y-b)/s,\quad \mu=b+s\widetilde\mu,\quad v=s^2\widetilde v,\quad
\mathrm{NLL}_y=\mathrm{NLL}_{\widetilde y}+\log s.$$

Training statistics yalnız training targets'tan gelir. Variance floor'un hangi koordinatlarda olduğu belirtilmelidir.

## 4. Ensemble predictive mixture

$$p_E(y|x)=\frac1M\sum_{m=1}^{M}p_m(y|x).$$

Classification: softmax distributions averaged. Regression: Gaussian mixture, genellikle tek Gaussian değil. Exact mixture log likelihood için log-sum-exp kullanılır; moment-matched Gaussian NLL ayrı ölçümdür.

## 5. Mixture mean, second moment ve total variance

$$\mu_E=\mathbb E_Z\mathbb E[Y|Z]=\frac1M\sum_m\mu_m,$$
$$\mathbb E[Y^2]=\frac1M\sum_m(v_m+\mu_m^2),$$
$$v_E=\mathbb E[Y^2]-\mu_E^2=\frac1M\sum_m v_m+\frac1M\sum_m(\mu_m-\mu_E)^2.$$

$$\operatorname{Var}(Y)=\mathbb E_Z\operatorname{Var}(Y|Z)+\operatorname{Var}_Z\mathbb E[Y|Z].$$

İlk parça within-member, ikinci parça between-member. Mixture'da divisor M kullanılır; unbiased sample variance düzeltmesi uygulanmaz. Aleatoric/epistemic yorumlar model varsayımlarına bağlıdır.

## 6. Prediction intervals ve coverage

$$I_z(x)=[\mu_E(x)-z\sqrt{v_E(x)},\;\mu_E(x)+z\sqrt{v_E(x)}].$$
$$\widehat C_z=\frac1{N_{obs}}\sum_j\mathbf1[y_j\in I_z(x_j)].$$

Ana toy plot z=3; ek noisy-target coverage z=1.95996398454. Diagnostic clean curve'de y_j yerine x_j³ kontrol edilir. Bunlar seed mean confidence interval'ı değildir.

## 7. FGSM ve random comparator

$$\ell(x+\delta,y)\approx\ell(x,y)+\delta^T\nabla_x\ell,\qquad |\delta_i|\leq\epsilon_i.$$
$$\delta_i^*=\epsilon_i\operatorname{sign}(\partial\ell/\partial x_i),\quad
L_{AT}=\ell(x,y)+\ell(x+\delta^*,y).$$

Coordinate-wise linear objective maksimumu gradient sign ile bulunur. Bizde epsilon training coordinate range'in .01 katıdır. Image input [0,1]'e clip edilir. +R'de sign yerine independent random ±1 yönü kullanılır; gradient bilgisi yoktur.

## 8. Softmax cross-entropy ve classification NLL

$$p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\quad CE=-\sum_k t_k\log p_k=-\log p_y.$$
$$\overline p_{nk}=\frac1M\sum_m p_{mnk},\quad P\in\mathbb R^{M\times N\times K},\quad
\overline P\in\mathbb R^{N\times K}.$$
$$\mathrm{NLL}=-\frac1N\sum_n\log\overline p_{n,y_n}.$$

Softmax nonlinear olduğundan logits ortalaması aynı estimator değildir. Implementation NLL'de 1e−12 target-probability floor kullanır; scoring float64 ile hesaplanır.

## 9. Paper-normalized Brier score

$$BS=\frac1{NK}\sum_{n,k}(p_{nk}-t_{nk})^2.$$
$$\mathbb E_q[BS(p,Y)|x]=\frac1K\left(\|p-q\|^2+1-\|q\|^2\right).$$

q'ya bağlı sabit dışında squared distance kalır; optimum p=q. K=10 normalizasyonu unutulursa ölçek on kat değişir.

## 10. Predictive entropy ve Jensen farkı

$$H(\overline p)=-\sum_k\overline p_k\log\overline p_k,\quad 0\log0=0.$$
$$H\left(\frac1M\sum_m p_m\right)\geq\frac1M\sum_m H(p_m).$$

İki farklı one-hot: member entropy=0, mixture probabilities (.5,.5,0,...) ve entropy=ln2. Entropy of average ile average entropy aynı değildir. Fark generalized Jensen-Shannon disagreement olarak yorumlanabilir; burada yeni deney sonucu olarak raporlanmaz.

## 11. Entropy bounds

$$D_{KL}(p\Vert u)=\sum_kp_k\log(Kp_k)=\log K-H(p)\geq0.$$
$$0\leq H(p)\leq\log K,\qquad K=10:\;H_{max}=\log10\approx2.302585\;\text{nat}.$$

Uniform maksimum, one-hot minimumdur. Classification entropy bound'u Gaussian differential entropy/NLL için geçerli değildir.

## 12. Entropy gap ve OOD AUROC

$$\Delta_H=\frac1{N_{OOD}}\sum H_{OOD}-\frac1{N_{ID}}\sum H_{ID}.$$
$$AUROC=\Pr(H_{OOD}>H_{ID})+\tfrac12\Pr(H_{OOD}=H_{ID}).$$

Positive class=OOD=1; ID=0. Aynı skorlar .5, perfect separation 1. Gap ortalama farkını; AUROC ranking'i ölçer. İşaret önceden tanımlıdır, kötü seed'i iyi göstermek için çevrilmez.

## 13. Üç seed'in mean ve sample SD'si

$$\overline a=\frac1S\sum_{s=1}^{S}a_s,\qquad
s_a=\sqrt{\frac{\sum_s(a_s-\overline a)^2}{S-1}},\qquad S=3.$$

ddof=1. Bu spread standard error (s/√S) veya confidence interval değildir. Ensemble üyeleri S olarak sayılmaz. Sabit data ile bu ölçü training run variation'ını gösterir.

## 14. MC Dropout estimator ve BatchNorm

$$\widehat p(y|x)=\frac1M\sum_{r=1}^M p(y|x;\theta,\omega_r),$$

Burada θ tek ağırlık seti, ω_r farklı dropout maskeleridir. Mask RNG stream'i bir kez seed edilir ve ilerler. BatchNorm evaluation mode ve sabit running buffers, dropout variation'ını training-statistic değişiminden ayırır.

## Referanslar

Raporun references.bib dosyası ve Section 2 ana kaynaktır: Lakshminarayanan et al. (2017), Gal and Ghahramani (2016), Goodfellow et al. (2015), Gneiting and Raftery (2007), Kingma and Ba (2015). Matematiği ezberlemeden önce her sembolün shape'ini ve olasılıksal varsayımını açıklayın.
