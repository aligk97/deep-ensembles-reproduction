# CMP4509 — Deep Ensembles / Toy Regression

[Deep Ensembles makalesinin Şekil 1'i](https://arxiv.org/abs/1612.01474) için PyTorch uygulaması. Aynı 20 gürültülü gözlem üzerinde sırayla 5 MSE üyesi, tek Gaussian NLL modeli, tek Gaussian NLL + FGSM modeli ve 5 Gaussian NLL üyesi eğitilir. Her ensemble üyesi aynı veriyle sıfırdan ve kendi optimizer ve RNG ayarlarıyla eğitilir.

## Hangi sürüm hangi dosyayı kullanıyor?

| Sürüm | Deneyi seçen ayar dosyası | Ne işe yarar? |
|---|---|---|
| V1 | [v1_baseline.yaml](src/toy_regression/configs/v1_baseline.yaml) | Başlangıç deneyini LR 0,001 ve batch 5 ile tanımlar. |
| V2-A | [v2_a_high_learning_rate.yaml](src/toy_regression/configs/v2_a_high_learning_rate.yaml) | Batch 5 iken öğrenme oranını 0,1’e çıkarır. |
| V2-B | [v2_b_full_batch.yaml](src/toy_regression/configs/v2_b_full_batch.yaml) | LR 0,001 iken 20 eğitim örneğinin tamamını tek batch’e alır (ayarda 100). |
| V2-C | [v2_c_high_learning_rate_full_batch.yaml](src/toy_regression/configs/v2_c_high_learning_rate_full_batch.yaml) | LR 0,1 ile tam batch’i birlikte kullanır. |

Bu dört deney aynı Python kodunu kullanır.

**Nereden başlamalıyım?** Eğitim akışı `run_training_experiment.py` → `experiment_commands.py` → `train_v1_v2_models.py` yolunu izler. Elindeki test tahminlerini değerlendirmek için `compare_regional_performance.py` ile başla. Bu komut tahminleri okur, metrikleri hesaplar ve rapor ile grafikleri yazar.

| Ortak kod dosyası | Görevi | Nerede kullanılır? |
|---|---|---|
| [run_training_experiment.py](src/toy_regression/run_training_experiment.py) | Seçilen YAML ile deneyi başlatır. | Terminalden eğitim başlatırken. |
| [experiment_config.py](src/toy_regression/experiment_config.py) | YAML ayarlarını okur ve parametreleri doğrular. | Eğitim komutunda, notebook’ta ve kayıtlı ayarları doğrularken. |
| [experiment_commands.py](src/toy_regression/experiment_commands.py) | Eğitim, tahmin ve kayıt adımlarını komut satırından yürütür. | Eğitim ve dışa aktarma girişleri ile notebook tarafından çağrılır. |
| [generate_cubic_data.py](src/toy_regression/generate_cubic_data.py) | Eğitim ve test verilerini üretir. | Her seed’in eğitiminden önce ve notebook önizlemesinde. |
| [v1_v2_regression_networks.py](src/toy_regression/v1_v2_regression_networks.py) | Regresyon ağlarının mimarisini tanımlar. | Eğitim bir ağ oluştururken veya analiz checkpoint yüklerken. |
| [regression_losses.py](src/toy_regression/regression_losses.py) | MSE, Gaussian NLL ve ensemble mixture NLL hesaplar. | Eğitimde optimizer adımında ve değerlendirmede NLL ölçerken. |
| [train_v1_v2_models.py](src/toy_regression/train_v1_v2_models.py) | Ağları eğitir ve checkpoint’leri kaydeder. | Eğitim komutu her seed için dört model koşulunu çalıştırırken. |
| [ensemble_predictions.py](src/toy_regression/ensemble_predictions.py) | Ağların tahminlerini ve ensemble varyansını hesaplar. | Eğitilmiş ağlardan test tahmini alırken veya checkpoint ile yeniden tahmin yaparken. |
| [evaluate_predictions.py](src/toy_regression/evaluate_predictions.py) | Her koşunun test hatasını, NLL ve kapsama değerlerini hesaplar. | Her model koşulunun sonunda ve bölgesel metriklerin aralık hesaplarında. |
| [plot_experiment_results.py](src/toy_regression/plot_experiment_results.py) | Eğitim eğrilerini, tahminleri ve belirsizlik bantlarını çizer. | Seed grafikleri, toplu grafikler ve notebook önizlemesinde. |
| [export_run_results.py](src/toy_regression/export_run_results.py) | Tamamlanmış bir koşunun sonuçlarını açar ve rapora aktarır. | Terminalden tamamlanmış bir koşunun tablosunu açarken veya rapora aktarırken. |
| [summarize_seed_results.py](src/toy_regression/summarize_seed_results.py) | Seed sonuçlarını birleştirip ortalama, standart sapma ve rapor tablosu üretir. | Üç seed’i özetlerken ve dışa aktarma komutunda. |
| [experiment_io.py](src/toy_regression/experiment_io.py) | Koşu dizinlerini, seed ayarlarını ve ortam kayıtlarını yönetir. | Eğitim, analiz ve arşivleme sırasında dizin ve kayıt işlemlerinde. |
| [archive_project_versions.py](src/toy_regression/archive_project_versions.py) | Kaynakları sürüm bilgisi ve dosya özetleriyle arşivler. | Eğitim ve analiz başında kodu saklarken veya terminalden proje arşivi alırken. |

| Bölgesel analiz kodu | Görevi | Nerede kullanılır? |
|---|---|---|
| [compare_regional_performance.py](src/toy_regression/compare_regional_performance.py) | Kayıtlı tahminleri sol OOD, eğitim aralığı ve sağ OOD için değerlendirir. | Terminalden kayıtlı deneyleri yeniden değerlendirirken. |
| [load_saved_predictions.py](src/toy_regression/load_saved_predictions.py) | Kayıtlı koşuları, tahminleri ve checkpoint’leri bulur. | Bölgesel karşılaştırma test girdilerini, tahminleri ve checkpoint’leri okurken. |
| [compute_regional_metrics.py](src/toy_regression/compute_regional_metrics.py) | Bölge bazında RMSE, NLL, kapsama, aralık genişliği ve varyans hesaplar. | Bölgesel karşılaştırma her model, seed ve bölge için skor üretirken. |
| [plot_regional_comparisons.py](src/toy_regression/plot_regional_comparisons.py) | Bölgesel karşılaştırmaları grafiklere dönüştürür. | Bölgesel karşılaştırmanın sonunda PNG ve PDF grafikleri oluştururken. |
| [write_regional_report.py](src/toy_regression/write_regional_report.py) | Türkçe analiz raporunu ve LaTeX tablolarını yazar. | Bölgesel karşılaştırmanın sonunda Markdown ve LaTeX yazarken. |

**Testler** `pytest`, [tests/](src/toy_regression/tests/) içindeki şu dosyaları çalıştırır.

- [test_cubic_data_and_config.py](src/toy_regression/tests/test_cubic_data_and_config.py) — Veri üretimi, test bölgeleri ve geçersiz ayarlar.
- [test_networks_and_losses.py](src/toy_regression/tests/test_networks_and_losses.py) — Ağ çıkış şekilleri, Gaussian ve mixture NLL.
- [test_adversarial_training.py](src/toy_regression/tests/test_adversarial_training.py) — FGSM girdileri ve adversarial eğitim koşulu.
- [test_ensemble_predictions.py](src/toy_regression/tests/test_ensemble_predictions.py) — Üye bağımsızlığı, ensemble varyansı ve MSE proxy’si.
- [test_regional_evaluation.py](src/toy_regression/tests/test_regional_evaluation.py) — Bölge sınırları, kapsama, RMSE ve eşleşen seed karşılaştırmaları.
- [test_training_run_archives.py](src/toy_regression/tests/test_training_run_archives.py) — Tamamlanan ve yarıda kalan eğitimlerin dosya kayıtları.

`src/toy_regression/` Şekil 1 deneyini, `src/mnist/` Şekil 2(a) için veri ve ağ kodunu içerir. `src/evaluation/` ortak değerlendirme çalışmaları için ayrılmıştır.

## MNIST Şekil 2(a) — Aşama 1

Depo kökünde, kullanacağınız Python ortamında aşağıdaki komutları çalıştırın. İlk komut kurulu torch ve torchvision sürümlerini ve CUDA durumunu gösterir. İki paket zaten çalışıyorsa mevcut kurulumu kullanın. Yeni ortam kurarken işletim sistemi ve NVIDIA sürücünüzle uyumlu torch ve torchvision çiftini resmi seçiciden yükleyin, ardından `python -m pip install -r requirements.txt` çalıştırın. [PyTorch kurulum seçicisi](https://pytorch.org/get-started/locally/)

```shell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m src.mnist.check_architecture --device cuda
python -m pytest -q src/mnist/tests
```

Denetim, torchvision ile resmi MNIST dosyalarını Git dışında tutulan `data/MNIST/` dizinine indirir. Sonraki çalıştırmalarda `--no-download` kullanılabilir. `--data-root` ile `MNIST/` klasörünün üst dizinini seçebilirsiniz. Varsayılan `--device auto`, CUDA varsa GPU'yu, aksi halde CPU'yu kullanır. Komut 60.000 eğitim ve 10.000 test örneğinin sayısını doğrular, her kümeden bir minibatch üzerinde değerlendirme modunda ileri geçiş yapar. Boyutları, etiketleri, piksel aralığını, parametre sayısını ve her görüntünün olasılık toplamını denetler.

Her başarılı çalıştırma [Aşama 1 doğrulama kaydını](results/metrics/mnist/phase1_sanity.json) günceller. Kayıtta UTC bitiş zamanı, yazılım ve donanım bilgileri, seed’ler, veri ve model boyutları ile her iki minibatch’in ölçümleri bulunur. Olasılık sapması, minibatch içinde bir görüntünün softmax toplamının 1’den en büyük mutlak farkıdır. Bellek değerleri bayt cinsindendir. CPU koşularında GPU tepe bellek ölçümü `null` olur. Bir kontrol başarısız olursa dosya son başarılı koşuyu göstermeye devam eder.

- **Makaleden alınan ayarlar.** Üç gizli katmanda 200'er birim, ReLU ve BatchNorm kullanılır. MC Dropout mimarisi için `--dropout-p 0.1` her gizli ReLU sonrasına dropout ekler. Batch boyutu 100, makalenin genel varsayılanıdır. [§3.1, §3.4 ve Şekil 2(a)](https://arxiv.org/pdf/1612.01474)
- **Uygulama seçimleri.** `ToTensor()` gri tonlu pikselleri [0, 1] aralığına taşır. Eğitim yükleyicisi her epoch'ta tüm örnekleri karıştırır, test yükleyicisi resmi sırayı korur. Gizli katmanlarda Linear → BatchNorm → ReLU sırası ve PyTorch'un varsayılan başlatması kullanılır. Ağın on logitlik çıktısı doğrudan `CrossEntropyLoss` için uygundur. Denetim, sınıf olasılıklarını görmek için çıkışa softmax uygular.
- **Tekrarlanabilirlik.** `--experiment-seed 42 --member-index 0` ilk üyeyi seçer. Depodaki seed kuralı model başlatması ve veri sırası için ayrı seed üretir. Python, NumPy ve PyTorch (CPU/CUDA) RNG'leri ayarlanır. Deterministik algoritmalar, sabit cuDNN davranışı ve bağımsız bir eğitim yükleyicisi üreteci kullanılır. Birebir tekrar kullanılan donanıma, PyTorch sürümüne ve deterministik çekirdek desteğine bağlıdır.
- **Kapsam.** Aşama 1 yalnızca veri ve mimari denetimidir. Sonraki aşama, tek model için kısa öğrenme oranı denemesi ve çapraz entropi ile eğitimdir. Aşama 1 çıktısı sınıflandırma başarımı ölçmez.

## MNIST Şekil 2(a) — Aşama 2, tek model

Depo kökünden yerel GPU eğitimini ve testleri çalıştırabilirsiniz. `--no-download` Aşama 1'de indirilen MNIST dosyalarını kullanır. İlk kurulumda veriyi indirmek için bu seçeneği kaldırın.

```powershell
python -m src.mnist.train_baseline --device cuda --no-download
python -m pytest -q
```

Deney önce Adam öğrenme oranları 0.1 ve 0.001 için aynı başlangıç ağırlıklarını ve aynı sıradaki 100 eğitim minibatch'ini kullanır. Tüm kayıpları sonlu ve son 20 batch'in ortalama kaybı ilk 20'den düşük olan adaylar arasından son ortalaması en düşük oran seçilir. Bu kısa karşılaştırma ve 10 epoch süresi yerel deney tercihleridir. Ardından ağırlıklar, Adam ve veri sırası sıfırlanır. Varsayılan dış seed 42, batch boyutu 100'dür. Eğitim 60.000 örneğin tamamını kullanır. Ayrı doğrulama kümesi ayrılmadığı için PNG, yalnızca eğitim kaybını gösterir. Her epoch değeri, o epoch'taki örnek sayılarıyla ağırlıklandırılmış minibatch kayıplarının ortalamasıdır.

Son epoch'tan sonra resmî test kümesindeki 10.000 örnek değerlendirilir. JSON'da `accuracy` kesir, `classification_error` yüzde, `nll` doğal logaritma ile örnek başına negatif log-olasılıktır. `brier_score`, tahmin ve tek-sıcak hedef arasındaki hata karelerinin önce 10 sınıf, sonra örnekler üzerinde ortalamasıdır. `training_time_seconds` CUDA senkronizasyonu dahil eğitim epoch sürelerinin toplamıdır. LR taraması (`probe_seconds`), checkpoint yazımı, çizim ve test değerlendirmesi bu sürenin dışındadır.

Her yeni çağrı ayrı bir koşu kimliği üretir.

| Dizin | Oluşan dosyalar |
|---|---|
| `results/metrics/mnist/<koşu>/` | `lr_batches.csv`, `lr_comparison.json`, `training_history.csv`, `test_metrics.json` |
| `results/raw/mnist/<koşu>/checkpoints/` | Her tamamlanan epoch sonunda güncellenen `baseline.pth` |
| `results/figures/mnist/<koşu>/` | `training_loss.png` |

Yarım kalan eğitime son tamamlanmış epoch'tan devam etmek için aynı komuta `--resume "results/raw/mnist/<koşu>/checkpoints/baseline.pth"` ekleyin ve `<koşu>` yerine gerçek klasör adını yazın. Tamamlanmış checkpoint ile bu komut eğitim adımı uygulamadan değerlendirmeyi ve grafiği yeniden üretir.

## MNIST — Ensemble öncesi epoch seçimi

Depo kökünde `python -m src.mnist.select_epochs` komutu önceden indirilmiş MNIST eğitim bölümünü kullanır. Seed 42 ile üretilen rastgele permütasyonun ilk 54.000 indeksi eğitime, kalan 6.000 indeksi validation'a ayrılır. İndekslerin 60.000 örneği tam bir kez kapsadığı denetlenir. Deney yalnızca bu bölümde çalışır.

Tek bir MNISTMLP, başlangıç ağırlıklarından itibaren CUDA üzerinde 20 epoch eğitilir. Adam öğrenme oranı 0.001, batch boyutu 100 ve dropout olasılığı 0'dır. Her epoch sonunda değerlendirme modunda validation accuracy (%), sınıflandırma hatası (%), NLL ve sınıf sayısına bölünmüş Brier skoru ölçülür. Seçim kuralı 10, 15 ve 20 arasından en düşük validation NLL'yi almaktır. Tam eşitlikte erken epoch seçilir.

Her çalıştırma ayrı bir koşu kimliği oluşturur. `results/metrics/mnist/<koşu>/` altında tüm epoch'lar (`epoch_history.csv`), üç aday (`epoch_comparison.csv`) ve karar (`epoch_selection.json`) bulunur. Dört PNG `results/figures/mnist/<koşu>/` altındadır. `results/raw/mnist/<koşu>/split_indices.npz` veri ayrımını, aynı dizinin `checkpoints/` klasörü ise üç adayın ağırlıklarını ve optimizer durumlarını içerir. Sonraki deneyler seçilen epoch sayısını kullanarak her modeli 60.000 eğitim örneğiyle sıfırdan eğitir.

## MNIST — Aşama 3, Standard Deep Ensemble

Komutları depo kökünde, Phase 2 ile aynı CUDA ortamında çalıştırın.

```powershell
python -m src.mnist.run_ensemble --smoke
python -m pytest -q src/mnist/tests src/toy_regression/tests
python -m src.mnist.run_ensemble --full
python -m src.mnist.run_ensemble --full --evaluate-only
```

Tam eğitim, 42, 123 ve 2026 dış seed'lerinin her biri için beş MNISTMLP'yi sırayla eğitir. Her üye yeni başlangıç ağırlıkları ve kendi eğitim yükleyicisi üreteciyle başlar. Resmî 60.000 örnek üzerinde epoch başına tek bir karıştırılmış geçiş yapılır. Adam LR 0.001, batch 100, dropout 0 ve 10 epoch kullanılır. Dış seed ve sıfır tabanlı üye indeksinden türetilen model ve karıştırma seed'leri ile başlangıç ağırlıklarının SHA-256 özeti kaydedilir. Tam deneyin girdisi, yerel `data/MNIST/` önbelleğidir.

Her epoch sonunda checkpoint atomik olarak güncellenir. Kesinti sonrasında aynı `--full` komutunu tekrar çalıştırın. Ayarları ve seed'leri eşleşen tamamlanmış Phase 3 üyeleri atlanır, yarım kalan üye son tamamlanan epoch'tan sürer. Uyumsuz checkpoint hata verir. Phase 2 ve epoch seçimi checkpoint'leri ayrı dizinlerdedir. `--evaluate-only` değerlendirme için tüm üyelerin tamamlanmış olmasını gerektirir.

Üyeler diskteki checkpoint'lerden birer birer GPU'ya yüklenir. Test örnekleri sabit sıradadır. Her dış seed için ilk 1, 3 ve 5 üyenin softmax sınıf olasılıklarının aritmetik ortalaması alınır. Sayısal kararlılık için toplam log uzayında hesaplanır. Accuracy [0, 1] aralığında, sınıflandırma hatası yüzde, NLL doğal logaritma ve Brier 1/10 sınıf normalizasyonuyla raporlanır. Üç dış seed üzerinden örneklem standart sapması kullanılır (ddof=1).

| Konum | Çıktılar |
|---|---|
| `results/raw/mnist/phase3_standard/checkpoints/seed_<seed>/` | Her üye için `member_1.pth` … `member_5.pth` ve aynı adlı CSV eğitim geçmişleri. Süre saniye, CUDA tepe belleği PyTorch'un ayırdığı bayt cinsindendir. |
| `results/metrics/mnist/phase3_standard/` | `experiment.json` ortam, ayarlar, seed'ler, üye süreleri ve bellek ölçümlerini içerir. `members.csv` tekil üye skorlarını, `seed_<seed>.json` ile `per_seed.csv` ensemble sonuçlarını, `mean_std.csv` ve `mean_std.json` seed'ler arası özeti içerir. |
| `results/figures/mnist/phase3_standard/` | Classification Error, NLL ve Brier skorunun M'ye göre değişimi, her biri PNG ve PDF. Hata çubukları ±1 örneklem standart sapmasıdır. |

Smoke modu aynı 15 üye ve M=1,3,5 akışını resmî bölümlerin ilk 200 eğitim ve 100 test görüntüsüyle, tek epoch çalıştırır. Tüm smoke çıktılarının klasör adı `phase3_standard_smoke` olur; bu ölçümler yalnızca işleyiş kontrolü içindir.

### V3 deneyleri

[toy_v3.yaml](src/toy_regression/configs/toy_v3.yaml) A–N deneylerinin batch, epoch, LR ve weight decay değerlerini tanımlar. Her ayarda üç seed ve seed başına 12 ağ eğitilir. N’de MSE ve Gaussian ağlarının öğrenme oranları farklıdır.

| Kaynak | Görevi |
|---|---|
| [run_v3_sweep.py](src/toy_regression/run_v3_sweep.py) | Deneyleri başlatır, verileri ve koşu kimliğini kaydeder. `--resume --run-id <kimlik>` ile yarım kalan eğitime devam eder. |
| [train_v3_members.py](src/toy_regression/train_v3_members.py) | Projenin mevcut MSE ve Gaussian ağlarını bağımsız minibatch sıralarıyla birlikte eğitir. Adam, üye, RNG ve epoch kayıtlarını yazar. |
| [verify_v3_checkpoints.py](src/toy_regression/verify_v3_checkpoints.py) | Üyeleri diskten yükler, tahminlerini yeniden hesaplar, son adımları ve veri hash’lerini kontrol eder. |
| [evaluate_v3_sweep.py](src/toy_regression/evaluate_v3_sweep.py) | Doğrulamayı çalıştırır, bölgesel ölçümleri toplar ve CSV ile validation.json oluşturur. |
| [report_v3_comparisons.py](src/toy_regression/report_v3_comparisons.py) | Aynı model ve seed için V1, V2 ve V3 farklarını hesaplar ve Türkçe raporu yazar. |
| [plot_v3_results.py](src/toy_regression/plot_v3_results.py) | Dört panelli Şekil 1, öğrenme eğrileri, kalibrasyon, genişlik ve bölgesel karşılaştırma grafiklerini kaydeder. |
| [v2_references_for_v3.yaml](src/toy_regression/configs/v2_references_for_v3.yaml) | V3 karşılaştırmasında kullanılan V2-A ve V2-C referanslarını ayrı bir V2 koşusunda tanımlar. |

Tam V3 eğitimi için `python -m src.toy_regression.run_v3_sweep --all --device cpu` kullanılır. `--smoke` kısa ön kontrolü ayrı bir koşunun `preflight/` dizinine yazar. Doğrulama ve rapor yolları [sonuç dizininde](results/README.md) belgelenmiştir.

### V4 öğrenme oranı deneyi

[v4_lr_0p3.yaml](src/toy_regression/configs/v4_lr_0p3.yaml) doğrulanmış V2-A ile aynı veri ve ağırlık başlangıçlarını kullanarak Adam LR’yi 0,3 yapar. Eğitim mevcut `run_v3_sweep.py --config` altyapısını kullanır. [evaluate_v4_lr_comparison.py](src/toy_regression/evaluate_v4_lr_comparison.py) iki koşunun checkpoint’lerini doğrular ve seed bazında karşılaştırır. [plot_v4_lr_comparison.py](src/toy_regression/plot_v4_lr_comparison.py) grafikleri, [report_v4_lr_comparison.py](src/toy_regression/report_v4_lr_comparison.py) Türkçe ölçüm tablolarını üretir. LR 0,03 için ayrı YAML ancak açıkça seçildiğinde çalışır.

## Yerelde çalıştırma

Komutları depo kökünden çalıştırın.

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
python -m src.toy_regression.run_training_experiment --help
python -m src.toy_regression.export_run_results --help
~~~

GPU'yu kontrol etmek, kısa bir deneme yapmak ve üç seed'li tam koşuyu başlatmak için aşağıdaki komutları kullanın. Kısa deneme isteğe bağlıdır.

~~~bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m src.toy_regression.run_training_experiment --smoke --device cuda
python -m src.toy_regression.run_training_experiment --device cuda
~~~

`--smoke` en fazla 3 epoch, 1 seed ve test kümesi başına en fazla 80 nokta kullanır. CPU ile smoke için `--device cpu` geçilebilir. `--device auto` CUDA varsa GPU kullanır. `--device cuda` CUDA kullanılamazsa açık hata verir. Windows'ta bu durumda önce NVIDIA sürücüsünü ve mevcut Python ortamındaki PyTorch paketinin CUDA desteğini kontrol edin. [PyTorch kurulum sayfası](https://pytorch.org/get-started/locally/)

Üç dış seed varsayılan olarak `42 123 2026`. `--seeds` ile geçersiz kılınabilir. Tam koşu, üye başına 3.000 epoch ile 36 model eğitir. Bu küçük MLP için süre donanım ve minibatch başına işlem maliyetine bağlıdır. Terminal ve `logs/run.log` üye bazında ilerlemeyi gösterir.

## Google Colab

1. [Colab](https://colab.research.google.com/) üzerinde [notebook'u](notebooks/01_toy_regression.ipynb) yükleyin veya GitHub üzerinden açın.
2. GPU koşusu için **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU** seçin. **Tümünü çalıştır** akışı kaynakları yerel çalışma dizininde arar ve gerekirse GitHub'daki `atahan` dalını klonlar. Uygulama henüz o dala aktarılmadıysa notebook, güncel çalışma kopyasını içeren bir ZIP yüklemenizi ister. ZIP'in kökünde veya tek bir üst klasöründe `src/` ve `requirements.txt` bulunmalıdır. Toy Regression ayarları `src/toy_regression/configs/` altındadır.
3. Ortam hücresinde `CUDA available: True` çıktısını kontrol edin. Notebook preset'i ve çözümlenmiş ayarları eğitim öncesinde gösterir.
4. Koşu tamamlandığında oluşan ZIP'i **Dosyalar** panelinden indirin. İsteğe bağlı Drive yedeği için eğitim başlamadan `BACK_UP_TO_DRIVE` seçeneğini işaretleyip hedef klasörü girin.

## Sonuç kaydı ve rapor

Her eğitim çağrısı, UTC zaman damgası ve yapılandırma hash'i içeren ayrı bir `results/raw/toy_regression/<RUN_ID>/` oluşturur. Çakışan adlara sıra numarası eklenir.

| Konum | İçerik |
|---|---|
| `manifest.json`, `config_resolved.yaml`, `environment.json`, `provenance.md` | Durum, çalıştırma komutu, Git revizyonu ve çalışma ağacı durumu, çözümlenmiş ayarlar, yazılım/donanım ve yöntem kaynakları |
| `seed_<SEED>/data/` | Eğitim, gürültülü iç/OOD testleri, temiz grid, veri RNG'leri ve gerçek FGSM epsilon |
| `seed_<SEED>/<METHOD>/` | Üye seed'leri, epoch CSV'leri, model/optimizer state dict'leri, üç tahmin NPZ'si ve metrikler |
| `seed_<SEED>/figures/` | Şekil 1, eğitim kayıpları, varyans ayrıştırması, üye tahminleri, kalibrasyon. 300 DPI PNG ve vektör PDF |
| `aggregate/` | Seed bazlı ve ortalama ± örneklem standart sapmalı metrikler, tüm seed'lerin paneli, tanısal karşılaştırma, CSV/LaTeX tablo, Türkçe bulgular ve şekil altyazısı |

 Tamamlanan koşuyu incelemek ve rapor için seçili çıktıları kopyalamak için `<RUN_ID>` yerine klasör adını yazın.

~~~text
python -m src.toy_regression.export_run_results --run-dir results/raw/toy_regression/<RUN_ID>
python -m src.toy_regression.export_run_results --run-dir results/raw/toy_regression/<RUN_ID> --export-report report/toy_regression
~~~

`--rebuild-to` ile toplu çıktıları kaydedilmiş seed metriklerinden yeni bir klasöre yeniden oluşturabilirsiniz. Arşiv değiştirilmez. Smoke ve üç seed'den az koşular rapor dışa aktarımına kapalıdır. Başarısız koşularda `failure.json` ve kısmi kayıtlar saklanır. Koşu kesilirse manifesti ve logu inceleyin. Sorun giderildiğinde bütün seed'leri yeni bir koşuda çalıştırın. [Rapor dosya yolları ve LaTeX örneği](report/toy_regression.md)

## Yöntem ve makaleden sapmalar

- DE2017 §3.2, 20 gözlemi ve `y=x³+ε` sürecini tanımlar. `ε` standart sapması 3, varyansı 9'dur. Uniform(-4,4) ve 100 birimli tek gizli katman, atıf verilen toy deneyinden gelir. [DE2017 §3.2 ve Şekil 1](https://arxiv.org/pdf/1612.01474v3), [HLA2015 §5.3](https://proceedings.mlr.press/v37/hernandez-lobatoc15.pdf)
- Pratik preset Adam LR 0.001, minibatch 5, 3.000 epoch ve PyTorch `Linear` başlatması kullanır. DE2017 §3.1 genel varsayılanları Adam LR 0.1, batch 100 ve Torch (Lua) başlatmasıdır. `src/toy_regression/configs/v2_c_high_learning_rate_full_batch.yaml` ilk iki varsayılanı tanısal olarak uygular. 20 örneğin tamamı aynı batch'e girdiğinde sıra değişimi yeni bir minibatch gradyan dizisi üretmez. Toy'a özgü epoch sayısı verilmemiştir. Her preset çözülmüş ayarları ve gerçek sapmaları arşivler. [DE2017 §3.1–3.2](https://arxiv.org/pdf/1612.01474v3)
- FGSM sadece üçüncü panelde çalışır. Epsilon her seed'in gözlenen x aralığının %1'idir. Girdiler kırpılmaz, hedefler ve native birimler korunur. Erken durdurma uygulanmaz. NaN/Inf hata üretir. Varsayılan gradient clipping kapalıdır.
- Şekil, 601 noktalı ortak grid ve ortak y ölçeği kullanır. Gri bant ortalama ±3 standart sapmadır. NLL ensemble bandı momentlerden, ana NLL tam Gaussian mixture yoğunluğundan ve kapsam mixture CDF kuantillerinden hesaplanır. Moment-matched NLL ve kapsam ayrıca kaydedilir. MSE satırının proxy NLL ve kapsamı `1e-6` varyans tabanını kullanır. Çizim ve ham tahminler ampirik varyansı korur.
- Gürültülü bağımsız testler, iç/OOD ayrımı ve üç-seed tablosu ek analizdir. OOD, x'in eğitim aralığı dışında olmasıdır. Dış seed deneyin tekrarıdır. Ensemble üye varyansı popülasyon varyansı, seed'ler arası standart sapma örneklem standart sapmasıdır (`ddof=1`). Tek modelde epistemic ve MSE ensemble'da aleatoric metrikleri N/A olarak kaydedilir. Test etiketleri ayar seçimi için kullanılmaz.
- `deterministic: true` desteklenen PyTorch deterministik işlemlerini etkinleştirir. CPU ve GPU arasında bit düzeyinde eşitlik beklenmez. YAML'da `batch_size: 20` veya `100` ile ayrı bir tam-batch duyarlılık koşusu yapılabilir.

Öğrenci katkıları ve AI araç kaydı [katkılar belgesinde](docs/contributions.md) tutulur. Eğitmenin [LaTeX şablonu](report/main.tex) raporun temelidir.
