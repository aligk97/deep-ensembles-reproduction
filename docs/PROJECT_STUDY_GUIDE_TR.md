# Proje çalışma kılavuzu

Bu kılavuzun amacı üç öğrencinin de bütün projeyi anlayabilmesidir. Sunum görevlerinin bölüşülmesi teoriyi bölüşmek anlamına gelmez. Önce [ana README](../README.md), sonra [nihai analiz](FINAL_RESULTS_ANALYSIS.md), [denklemler](KEY_EQUATIONS.md) ve [savunma soruları](DEFENSE_QUESTIONS_AND_ANSWERS.md) okunmalıdır.

## 1. Büyük resim: neyi reproduce ettik?

Lakshminarayanan, Pritzel ve Blundell'in 2017 Deep Ensembles makalesindeki üç seçilmiş karşılaştırmayı ele aldık: Figure 1 Toy Regression, Figure 2(a) MNIST Classification, Figure 3(a) MNIST/NotMNIST OOD. Makalenin regression benchmark, SVHN, ImageNet veya confidence-threshold deneylerini yaptığımızı iddia etmiyoruz. Proje yalnız başarı raporu değildir: toy sol OOD başarısızlığı ana sonuçtur.

Ana sonuçların tamamı gerçek Colab Tesla T4 koşularındandır. Yerel sonuçları ek reproducibility karşılaştırması olarak kullanırız. Bir yöntemi daha iyi göstermek için ortam veya seed seçmeyiz. Son teslim hazırlığında yeni training, hyperparameter search veya model inference yapılmadı; CSV/JSON/manifest/figure/notebook evidence okundu.

## 2. Dağılım tahmini neden önemlidir?

Point prediction tek bir değer verir. Predictive distribution olası hedeflerin nasıl dağıldığını söyler. Classification'da çıktı on sınıflı probability vektörüdür; regression'da ağ mean ve variance üretir. Accuracy probability değerlerini dikkate almaz: doğru sınıfa .51 veya .99 vermek aynı argmax'ı oluşturabilir. NLL ve Brier bu farkı değerlendirir.

Proper scoring rule, population düzeyinde doğru dağılımı bildirmeyi ödüllendirir. Model kısıtı, az veri, optimization ve distribution shift varken training loss'un bu özelliği calibration garantisi değildir. Bu ayrımı özellikle toy başarısızlığını açıklarken koruyun.

## 3. Toy Regression akışı

- Aynı 20 satırlık `toy_regression/outputs/toy_training_data.csv` bütün run'larda kullanılır. Data seed 42'dir; run seed'leri 42,123,2026'dır. Aynı sayı farklı amaçlarla kullanılabilir.
- Generative process: x uniform [-4,4], y=x³+Gaussian noise, true noise SD=3. Grid [-6,6]'da 400 x değeridir. Her x için 20 yeni noisy target vardır; evaluation seed 20261122 sabittir.
- MSE modelleri 1–50–ReLU–1, Gaussian modelleri 1–50–ReLU–2'dir. Adam: MSE .01, Gaussian .001; 3,000 full-batch epoch. Gaussian targets training mean/scale ile normalize edilir.
- Dört yöntem: beşli MSE Ensemble; tek Gaussian NLL; tek Gaussian NLL+AT; beşli Gaussian Deep Ensemble. Sonuncusu burada AT içermez.
- Gaussian raw variance softplus+1e−6 ile pozitiftir. Original y ölçeğine mean = offset+scale×normalized mean, variance = scale²×normalized variance olarak dönülür.
- Member seed kuralı bu deneyde `100*run_seed + member_index`tir. MNIST kuralıyla karıştırmayın.

MSE variance yalnız üyelerin mean disagreement'ından gelir. Gaussian ensemble'da mean member variance ve variance of member means ayrı hesaplanır. Modelin aleatoric/epistemic yorumları varsayımlara bağlıdır; her üye aynı yanlılığı paylaşabilir.

Ana Figure 1 tarzı paneller ±3σ kullanır. `coverage95` ise ±1.95996398454σ aralığında **yeni noisy observations** bulunma oranıdır. Diagnostic `clean_curve_coverage_95`, noiseless x³ için farklı bir sorudur. Üç seed'in skor SD'si ise bu prediction interval'ların hiçbiri değildir.

En önemli olumsuz bulgu: sol OOD'de dört yöntemin üçer seed'inde coverage sıfırdır; hem yerelde hem T4'te. Colab Deep Ensemble clean RMSE'si ID'de 5.063±.109, solda 79.728±.454'tür. Ortalama öğrenilmiş variance'ın floor'a yaklaşması ve ortak ReLU extrapolation bias olası açıklamalardır; nedensel ablation kanıtı değildir. Exact mixture NLL ve moment-matched Gaussian NLL kolonlarını ayrı okuyun; büyük NLL'leri gizlemeyin.

**Pratik:** `07_seed_experiments/aggregate/metrics/metrics_by_seed.csv` içinden seed 42/deep_ensemble/left_ood satırını bulun. Hangi target, region, interval ve variance approximation kullanıldığını kendi cümlelerinizle açıklayın. Ardından üç seed'den sample SD'yi hesaplayın.

## 4. MNIST Classification akışı

MNIST official training images 60,000'dir. Split seed 4509 ile 55,000 training ve 5,000 validation ayrılır. Test 10,000 görüntüdür. Preprocessing uint8/255 → float32 [0,1], 784 features; ek mean/std normalization yoktur. Validation index'leri training'e girmez; epsilon range de training subset'ten hesaplanır.

Network 784–200–200–200–10'dur. Üç hidden layer'da ReLU ve BatchNorm vardır. Toplam 240,610 trainable parameter bulunur. MC Dropout, her ReLU sonrasına p=.1 ekler. Final profile 10 epoch, Adam lr=.1, batch size=100, evaluation batch size=1000 kullanır. Final epoch değerlendirilir; test skoruna bakarak checkpoint seçilmez. Eski tuning ayrıntıları son arşivlerde bütünüyle yoktur; yapılmış gibi validation search anlatısı uydurmayın.

MNIST initialization kuralı non-dropout için `10000*run_seed + member + 1`, dropout için `10000*run_seed + 9001`dir. Ensemble/+R/+AT paired başlangıç seed'lerini paylaşır, fakat ayrı modeller olarak farklı training objective'lerle eğitilir. Shuffle generator ayrıdır. +R random sign, +AT gradient sign pertürbasyonu üretir; clean ve perturbed CE loss toplanır.

48 model = üç seed × (5+5+5+1). M=1..5 skorları aynı beşli probability stack'in prefix'lerinden hesaplanır; dropout'ta M stochastic forward sayısıdır. 63 CSV satırı, seed başına 20 karşılaştırma ve aynı Ensemble M=1'i temsil eden ek single referanstan gelir. 63 ayrı eğitim yoktur.

Probabilities [M,N,10] şeklindedir; önce M boyunca ortalanır. Accuracy argmax, NLL −log(p_true), Brier örnekler ve 10 sınıf üzerindeki squared-error mean'dir. NLL sayısal olarak 1e−12 floor kullanır; metrikler float64 ile hesaplanır. Brier'daki 1/K unutulursa sayı on kat değişebilir.

M=5 Colab accuracy: Ensemble %98.54±.11, +R %98.47±.21, +AT %98.61±.09, MC Dropout %97.32±.22. Bütün M değerlerinde monoton iyileşme yoktur: AT error M4→M5'te %1.363→%1.390, dropout %2.670→%2.680. Bunları silmeyin; küçük farklardan significance sonucu çıkarmayın.

**Pratik:** `engine.py` içindeki `training_step`, `predict` ve `evaluation_mode` çağrılarını izleyin. Ardından seed/method/member/epoch kaydını manifest ve audit ile eşleştirin. Bir classification_error sayısını yüzdeye çevirirken neden 100 ile çarptığınızı açıklayın.

## 5. OOD akışı ve checkpoint bağımlılığı

OOD yeni eğitim yapmaz. Önce tamamlanmış Figure 2(a) Colab run'ı gerekir. Drive dizini `DeepEnsembles_MNIST_Figure2a/<MNIST_RUN_ID>/mnist_classification/outputs/local` içindedir. OOD notebook'un `MNIST_RUN_ID` değeri bununla aynı olmalıdır; ayrı `OOD_RUN_ID` çıktı izolasyonunu sağlar. Tam MNIST Colab export ZIP'i açıkça alternatif source seçilebilir. Eksik veya yanlış kaynakta durulur; Windows modeline otomatik geçilmez.

NotMNIST small arşivinde 18,726 PNG vardır; iki known corrupt dosya elenince 18,724 kalır. MNIST 10,000 test örneğinin tamamı ve NotMNIST'in bütün geçerli örnekleri kullanılır. Görüntüler native grayscale/light-on-dark, 28×28, float32/255, 784 features'tır. Harf label'ları metadata'dır; digit accuracy hedefi yapılmaz.

Windows ve Linux `sorted(Path)` karşılaştırması farklıdır. Aynı data'nın 16,719 konumu farklı sıralanmıştı. `colab_data.py`, relative class/filename sırasını case-insensitive referansa alarak images, targets ve paths'i birlikte permüte eder. Native ve aligned tensor hash kaydedilir, aligned tensor ve order hash yerel referansla eşleşmek zorundadır. Bu düzeltme pixel transform veya dataset değişikliği değildir. MC Dropout stream'inin hangi görüntüyle hangi maskeyi eşleştirdiğini de sabit tutar.

MC Dropout'ta yalnız dropout aktif kalır. BatchNorm running buffers sabittir. RNG her pass öncesi tekrar seed edilmez; bir stream başlangıcında seed edilip beş farklı pass boyunca ilerler. Aksi halde aynı maskeleri tekrar üretmek sahte bir ensemble oluşturabilir.

36 seed satırı = 3×4×3; 12 summary = 4×3. Cache sayıları: 24 probability grup = 3 seed×4 yöntem×2 data; 72 entropy array = aynı grup×3 M. Cache bir checkpoint değildir: checkpoint model ağırlıklarını; cache hesaplanmış probabilities/entropy'yi saklar. Source/config/data/runtime imzası değişince cache karıştırılmaz.

Entropy H=−Σp_bar ln p_bar; unit nat, K=10 için [0,ln10]. Gap=mean OOD−mean ID. AUROC'da ID=0/OOD=1 ve yüksek entropy yüksek OOD score'dur. M=5 Colab AUROC: Ensemble .9653±.0038, +R .9346±.0118, +AT .9668±.0023, dropout .7779±.0736. AUROC/gap paper'ın Figure 3(a) histogramına ek analizlerdir. Yerel Ensemble M1/seed123 AUROC .3790 olumsuz sonucu korunur. Yüksek ortalama AUROC bütün OOD örneklerinin güvenli olduğu anlamına gelmez.

**Pratik:** OOD `classifier_reference/manifest.json` ile MNIST manifestinin hash'ini karşılaştırın. Inventory'deki 48 SHA'nın audit ile nasıl eşleştiğini gösterin. Bir one-hot ve bir uniform input probability tensor'u için entropy'yi elle hesaplayın.

## 6. Sonuçları savunurken doğru cümleler

- “Bu protokolde ve ölçülen M aralığında ensemble-versus-dropout nitel eğilimi desteklendi.”
- “Toy Regression kısmi reproduction; sol OOD belirsizliği yetersiz kaldı.”
- “Üç seed'in sample SD'sini raporladık; significance testi yapmadık.”
- “Bütün primary sonuçlar Colab'dan; yerel sonuçlar supplementary karşılaştırma.”
- “Shared bias ve variance collapse olası açıklamalar, kanıtlanmış nedenler değil.”
- “Makalenin bütün deneylerini veya universal calibration'ı doğrulamadık.”

## 7. Kod ve kanıt okuma haritası

| Konu | Önce okunacak dosya | Kanıt |
|---|---|---|
| Toy objective/mixture | `toy_regression/07_multi_seed_experiments.py` | `results/colab/toy_regression/07_seed_experiments/aggregate/metrics/` |
| Toy failure diagnosis | `toy_regression/08_diagnose_uncertainty.py` | `results/colab/toy_regression/08_uncertainty_diagnostics/` |
| Classifier model/modes | `mnist_classification/model.py` | checkpoint audit + actual executed notebook |
| Training and perturbations | `mnist_classification/engine.py` | final config, histories in original export, summary CSV |
| Probabilistic scores | `mnist_classification/metrics.py` | `results/colab/mnist_classification/raw_results.csv` |
| OOD data/ordering | `mnist_notmnist_ood/data.py`, `colab_data.py` | dataset_info + order_alignment + test reports |
| OOD mode/RNG/probabilities | `mnist_notmnist_ood/inference.py` | inference_test_results + experiment_manifest |
| Predictive entropy/AUROC | `mnist_notmnist_ood/metrics.py` | 36 raw / 12 summary rows |
| Delivery numeric checks | `scripts/validate_delivery.py` | FINAL_VALIDATION_REPORT |

## 8. Üç kişi için çalışma planı

Her üye önce aynı üç teori türetimini boş kağıtta yapsın. Sonra sırayla bir deney anlatırken diğer ikisi soru sorsun; bir sonraki provada rolleri değiştirin. En son herkes rastgele seçilmiş bir dosya/CSV satırının config, seed, checkpoint ve figure provenance'ını izlesin. Sunumda Student 1/2/3 akışı yalnız süre yönetimi önerisidir; gerçek katkı tablosunu bu öneriden kopyalamayın.

## 9. Teslim öncesi insan kontrolü

İsimler, öğrenci numaraları, gerçek katkılar ve AI disclosure gözden geçirilmeli. Dersin sonradan ilan edilen sayfa sınırı ve atanan sunum günü kontrol edilmeli. Mevcut public repository URL gerçek, fakat final temiz içerik kullanıcı onayından sonra yayınlanmalı. Güncel PDF yeniden derlenip bütün sayfaları açılmalı. Sunum sesli prova edilip yaklaşık 7:30 anlatım + 2:30 soru-cevap sınırına uydurulmalı. Ders yönergesi slayt/not okumayı yasaklar; scripts yalnız prova içindir.
