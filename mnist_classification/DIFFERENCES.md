# Makale - uygulama karşılaştırması

Kaynak: yerel `1612.01474v3.pdf`, arXiv v3 (4 Kasım 2017), §2.3–2.4, Algorithm 1, §3.1 ve §3.4. Genel hiperparametreler §3.1'de belirtilmiştir; lr=0.1 ve batch=100'ün makalede verilmediği ileri sürülmemelidir. Ders yönergesi daha küçük hesaplama bütçesini açıklamak koşuluyla kabul eder.

| Bileşen | Makale | Bu uygulama / gerekçe |
|---|---|---|
| Ağ | 3 gizli katman, 200 ünite, ReLU, BN | Aynı boyutlar; 784-200-200-200-10 |
| BN yerleşimi | Tam sıra açıklanmıyor | Linear -> BN -> ReLU; yaygın, açık uygulama tercihi |
| BN ayarları | Ayrıntı verilmemiş | PyTorch eps=1e-5, momentum=0.1, affine=True |
| MC dropout | Her nonlinearity sonrası 0.1 | Aynı; testte yalnız dropout aktif, BN sabit |
| Optimizasyon | Adam, sabit lr=0.1, batch=100 | Korundu; betas, Adam eps ve weight decay PyTorch varsayılanları, ek scheduler yok |
| Başlangıç | Torch varsayılanı | PyTorch varsayılanı; tarihsel Torch ile sayısal eşitlik iddiası yok |
| Eğitim uzunluğu | MNIST epoch/step sayısı açık verilmemiş | 10 epoch sabit bütçe; smoke 1 epoch. Yakınsama garantisi yok |
| Veri | Standart MNIST train/test | Resmi 60k/10k korunur; 60k içinden sabit 5k validation ayırdığımız için asıl eğitim 55k. Test hiperparametre seçiminde kullanılmaz |
| Veri işleme | MNIST piksel ölçekleme ayrıntısı açık değil | /255; z-score yok; train/test birleştirilmez; crop/flip yok |
| Veri kütüphanesi | Tarihsel Torch veri yolu | torchvision yerine MD5 kontrollü IDX okuyucu; aynı örnekler, daha az ikili bağımlılık |
| Epsilon | Her boyutun eğitim aralığının %1'i | Aynı: 0.01*(max_train_j-min_train_j); sabit aralıklı piksellerde 0 |
| FGSM | x + epsilon sign(grad CE) | Aynı yön; ayrıca [0,1]'e clip. Clip açık bir uygulama farkıdır, piksel alanını korur |
| Random | Rastgele signed vector | Her batch/örnek/piksel için bağımsız ±1; AT ile aynı epsilon ve clip |
| AT/R kaybı | Clean + perturbed loss | Toplam CE; 1/2 katsayısı yok. Her batch'te iki BN güncellemesi, temiz örneklerin ek gradient forward'ı yeniden kullanılacak şekilde uygulanır |
| Çeşitlilik | Random initialization, veri shuffle; bagging yok | Her üye ayrı seed, optimizer ve shuffle. Veri bölümü üyeler arasında aynı |
| Tedaviler arası seed | Açıklanmıyor | Clean/R/AT aynı üye indeksinde eşleştirilmiş başlangıç seed'i. Yöntemlerin ağırlıkları paylaşılmaz |
| M | Figure 2(a) 15'e kadar | full konfigürasyon: 1,2,3,4,5,10,15; local: 1–5. Gerçek tamamlanan kapsam manifest ve sonuç raporunda |
| MC karşılaştırması | M stochastic dropout samples | Bir seed'de bir dropout ağı, M geçiş; M ayrı dropout ağı değil |
| Single baseline | Ensemble M=1 | İlk clean üye aynı sonucu temsil eder; gereksiz yeniden eğitim yok |
| Tahmin | Olasılıkların eşit ağırlıklı karışımı | Aynı; softmax sonrası ortalama |
| NLL | -log p(y|x) | Aynı; FP32 softmax, FP64 skor; log öncesi 1e-12 taban sayısal tercih |
| Brier | 1/K sum_k (t_k-p_k)^2 | Aynı, sonra N test örneğinde ortalama; K=10 |
| Tekrarlar | Figure 2(a)'nın seed/SD protokolü açık değil | 42,123,2026; sample SD ddof=1; ders gereği. Noktalarda error bar eklendi |
| M değerlendirmesi | M'ye göre metrik eğrileri | Her seed'de sabit üye sırasının ilk M'si; nested tahminler. En iyi altküme seçimi yok |
| Model seçimi | MNIST ayrıntıları açık değil | Her zaman son epoch; validation yalnız izleme, test sadece eğitim sonrası |
| Çalışma ortamı | Tarihsel Torch | PyTorch FP32, deterministic CUDA, AMP/TF32 kapalı; Windows yerel GPU. Colab uyumlu kod, gerçek Colab yürütmesi ayrıca belirtilmeli |
| Ek çıktılar | Figure 2(a), nitel örnekler | PNG/PDF, seed CSV, training/validation geçmişi, sınıf olasılıkları, confidence örnekleri. Güven örnekleri seed=42 AT modelinden sabit M ile üretilir; test üzerinde model seçimi değildir |

## Yorum sınırları

Makale eğrileri sayısal ground truth olarak veri dosyalarına girilmez. Yaklaşık eğilim karşılaştırması yapılır; özgün MNIST ham sonuçları elimizde değildir. Üç seed geniş bir belirsizlik analizi sağlamaz; seed'ler üzerindeki standart sapma aynı sabit test setindeki eğitim rastlantısallığını ölçer. Error bar örtüşmesi/örtüşmemesi tek başına istatistiksel anlamlılık testi değildir.

AT'nin R'den veya clean eğitimden her koşulda daha iyi olması zorunlu değildir. Test sonucunu görünce lr/epoch/epsilon değiştirilmez. Bu kurulumda ana iddianın hangi kısmının desteklendiği ve hangisinin desteklenmediği gerçek sonuç raporunda ayrı yazılır. MNIST confidence görsellerinden OOD davranışı hakkında sonuç çıkarılmaz.
