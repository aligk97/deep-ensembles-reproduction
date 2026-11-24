# CMP4509 - MNIST Deep Ensembles yeniden uygulaması

Hedef: Lakshminarayanan, Pritzel ve Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, Figure 2(a). [Makale](https://arxiv.org/abs/1612.01474), özellikle §2.3, Algorithm 1, §3.1 ve §3.4. Yerel kaynak: `../1612.01474v3.pdf`. Ders yönergesi: `../CMP4509_Project_Announcement.pdf`.

Bu klasör bağımsızdır; `toy_regression` dosyalarına yazmaz. Kod gerçek deney üretir; makalenin eğrilerinden sayısal sonuç kopyalamaz. Sonuçların kapsamı için `outputs/<profil>/manifest.json`, `summary.csv` ve `SONUC_RAPORU.md` okunmalıdır. Smoke sonuçları bilimsel sonuç olarak kullanılmamalıdır.

## Hızlı başlangıç (Windows PowerShell)

Ana proje klasöründe:

```powershell
python -m venv mnist_classification/.venv
# NVIDIA GPU için resmi PyTorch kurulum seçicisine uygun CUDA wheel kullanın.
# Bu bilgisayarda test edilen kurulum:
mnist_classification/.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
mnist_classification/.venv/Scripts/python.exe -m pip install -r mnist_classification/requirements.txt
mnist_classification/.venv/Scripts/python.exe mnist_classification/test_correctness.py
mnist_classification/.venv/Scripts/python.exe mnist_classification/run_experiment.py --config mnist_classification/configs/smoke.json
mnist_classification/.venv/Scripts/python.exe mnist_classification/run_experiment.py --config mnist_classification/configs/local.json
```

Linux / Colab terminalinde aynı komutlarda sanal ortam yolu `.venv/bin/python` olur. CPU için PyTorch CPU paketi ve `--device cpu` kullanılabilir. `--device auto` varsayılandır. `torchvision` gerekli değildir: MNIST'in resmi IDX dosyaları doğrudan okunur ve bilinen MD5 değerleriyle doğrulanır. İnternet yalnızca ilk veri/paket indirmesinde gerekir.

Geniş kapsam:

```powershell
mnist_classification/.venv/Scripts/python.exe mnist_classification/run_experiment.py --config mnist_classification/configs/full.json
```

| Profil | Eğitim / doğrulama / test | Epoch | M | Bağımsız eğitim sayısı |
|---|---|---:|---|---:|
| smoke | 1.000 / 5.000 / 200 | 1 | 1, 2 | 21 |
| local | 55.000 / 5.000 / 10.000 | 10 | 1, 2, 3, 4, 5 | 48 |
| full | 55.000 / 5.000 / 10.000 | 10 | 1, 2, 3, 4, 5, 10, 15 | 138 |

Tüm profillerde deney seed'leri **42, 123, 2026**. `single`, temiz ensemble'ın ilk üyesidir; ayrıca aynı modeli eğitmek gereksizdir. MC Dropout her seed için tek model eğitir, M stokastik geçiş yapar. Bir M konfigürasyonu için sonuçları ayrı ayrı eğitim yapmak yerine ilk M üyeyi kullanarak hesaplamak hem yöntem açısından doğrudur hem maliyeti azaltır. Farklı M sonuçları dolayısıyla bağımsız örnekler değildir.

## Google Colab

`MNIST_Deep_Ensembles_Colab.ipynb` dosyasını Colab'a yükleyin, **Runtime > Change runtime type > T4 GPU** seçin ve hücreleri sırayla çalıştırın. Notebook gerekli kaynak kodunu içinde taşır; GitHub adresi veya dosya yüklemesi gerektirmez. Önce doğruluk testlerini ve smoke testini, sonra seçili ana profili çalıştırır. Varsayılan ana profil `local`; tüm M değerleri için `PROFILE = "full"` seçilir. `USE_DRIVE = True` kesinti sonrası checkpoint'lerin Drive üzerinde kalmasını sağlar; hesabınıza bağlanmanız gerekir. GPU'nun ücretsiz hesapta tahsis edilmesi platformun müsaitliğine bağlıdır.

Colab'da önceden bulunan PyTorch kullanılır. Aynı matematiksel yöntem, paket/GPU farkları nedeniyle bit düzeyinde aynı sayıları garanti etmez. Notebook yerel olarak smoke modunda baştan sona doğrulanmıştır; gerçek Colab oturumu ayrıca çalıştırılmadıysa raporda Colab üzerinde çalıştırıldı denmemelidir. Notebook kaynaklarını yenilemek: `python mnist_classification/build_notebook.py`.

## Yöntem ve veri disiplini

- Ağ: **784 -> [Linear(200), BatchNorm, ReLU] x 3 -> Linear(10)**. MC Dropout'ta her ReLU'dan sonra `Dropout(0.1)`. Model logits döndürür; eğitim `cross_entropy` ile yapılır.
- MNIST resmi 60.000 eğitim ve 10.000 test örneği ayrı tutulur. Eğitim bölümü sabit seed **4509** ile 55.000 eğitim / 5.000 doğrulamaya ayrılır; tüm deneylerde aynı bölme kullanılır. Test setinden model, epoch veya hiperparametre seçilmez.
- Piksel değerleri 255'e bölünür, [0,1] aralığında kalır; crop/flip, z-score normalizasyonu ve bagging yoktur. Epsilon yalnızca kullanılan eğitim alt kümesinin piksel aralıklarından hesaplanır.
- **Single / Ensemble:** temiz örneklerde CE. Üyeler bağımsız başlangıç ve minibatch sırası kullanır.
- **Random (R):** bağımsız Rademacher yönü, `x_r = clip(x + epsilon * r, 0, 1)`, `r_j in {-1,+1}`; amaç `CE(x,y) + CE(x_r,y)`.
- **Adversarial (AT):** `x_adv = clip(x + epsilon * sign(grad_x CE(x,y)), 0, 1)`; amaç `CE(x,y) + CE(x_adv,y)`. Perturbation detach edilir, ikinci mertebe türev hesaplanmaz.
- **MC Dropout:** testte önce `model.eval()`, ardından yalnızca `nn.Dropout.train()`. BatchNorm running mean/variance ve `num_batches_tracked` değişmez. M, bağımsız model sayısı değil dropout geçiş sayısıdır.
- **Olasılık karışımı:** `p_bar(k|x) = (1/M) sum_m softmax(f_m(x))[k]`. Logits ortalaması alınmaz.

Üye seed'i `run_seed * 10000 + member_index + 1`; MC model seed'i `run_seed * 10000 + 9001`. Üç ensemble tedavisinde aynı üye sırası eşleştirilmiş başlangıçlar kullanır. Her tedavi kendi parametrelerine, optimizer'ına ve RNG akışına sahiptir; ağırlık paylaşımı yoktur. Shuffle generator seed'i üye seed'i + 100.000.000; MC test RNG seed'i MC model seed'i + 500.000.000. Üyeler deney tekrarı sayılmaz; standart sapma üç run seed üzerinden hesaplanır.

## Eğitim ayarları

| Ayar | Değer / kaynak |
|---|---|
| Optimizer | Adam; makale §3.1 |
| Learning rate | sabit **0.1**; makale §3.1, test üzerinde ayarlanmadı |
| Batch size | **100**; makale §3.1 |
| Epoch | **10** (smoke: 1); bizim bütçe tercihimiz, makalede MNIST için verilmemiş |
| Epsilon | `0.01 * (train_pixel_max - train_pixel_min)`; makale §3.1 |
| Adam betas / epsilon / weight decay | (0.9, 0.999) / 1e-8 / 0; uygulama varsayılanı, makalede açık değil |
| BatchNorm | affine=True, eps=1e-5, momentum=0.1; PyTorch varsayılanları |
| Başlangıç | PyTorch Linear / BatchNorm varsayılanı |
| Dropout | gizli ReLU'lardan sonra p=0.1; makale §3.4 |
| Son model | son epoch; early stopping / test tabanlı seçim yok |
| Sayısal işlem | FP32 eğitim; deterministic algoritmalar; TF32 ve AMP kapalı |

AT ve R modellerinde her batch için temiz ve perturbed forward yapıldığı için BatchNorm istatistikleri iki kez güncellenir. Validation çalışması deterministic dropout-kapalı modelle izlenir; checkpoint seçimi için kullanılmaz. Yüksek öğrenme oranı nedeniyle eğitim dalgalanabilir; sonuçlara bakıp sessizce lr düşürülmez.

## Metrikler

`N` test örneği, `K=10`, `t_nk=1[y_n=k]`:

```text
accuracy = mean_n 1[argmax_k p_bar_nk = y_n]
classification_error = 1 - accuracy
NLL = -mean_n log(max(p_bar_n,y_n, 1e-12))
Brier = (1/(N*K)) sum_n sum_k (p_bar_nk - t_nk)^2
```

CSV accuracy/error değerleri [0,1] oranıdır; grafikte error yüzdeye çevrilir. NLL doğal logaritma ile nats cinsindedir. Brier score bu makalede **1/K** içerir; normalizasyonsuz bazı kütüphanelerin sonucunun 10 katı olmasına dikkat edin. Skorlar FP64 toplanır; olasılıklar FP32 softmax'tan gelir. NLL'deki 1e-12 tabanı aşırı küçük olasılıklar için sayısal bir tercihtir ve NLL'yi sınırlar. `summary.csv` değerleri üç seed'in aritmetik ortalaması ve **sample standard deviation (ddof=1)** değeridir; error bar confidence interval veya standard error değildir.

## Checkpoint ve yeniden başlatma

Aynı komutu ve aynı çıktı klasörünü yeniden çalıştırın. Bitmiş üyeler yeniden eğitilmez; yarım kalan üye son tamamlanan epoch'tan sürer. Checkpoint model/BN buffer'larını, Adam durumunu, epoch'u, history'yi, Python/NumPy/CPU/CUDA/shuffle RNG durumlarını içerir. Dosya geçici yola yazılıp atomik replace edilir. En fazla devam eden epoch kaybolur. Diskte sadece son checkpoint tutulur. CPU ve CUDA (varsa) üzerinde kesintisiz ve 1. epoch sonrası yeniden başlatılan eğitimlerin **tüm dört yöntem için bit düzeyinde eşitliği test edilir**; farklı donanıma geçiş bu garantiye dahil değildir.

Konfigürasyon veya temel eğitim kaynakları değişirse eski checkpoint'in yanlışlıkla kullanılması SHA256 imzasıyla engellenir. Böyle bir değişiklikte yeni `--output` yolu verin. Aynı CUDA donanımı ve yazılım yığınıyla sürdürün; eski checkpoint'e farklı sürüm yüklemek önerilmez. Eğitimler tamamlandıktan sonra test değerlendirmesi başlar. Değerlendirme kesilirse yeniden çalıştırma eğitimleri atlar, test tablolarını deterministik sırayla yeniden hesaplar. Yerel `.pt` dosyaları optimizer/RNG nesneleri içerdiğinden güvenilir kaynak varsayılır; yabancı checkpoint yüklemeyin.

Sadece mevcut kayıtlardan grafikleri yeniden üretmek:

```powershell
mnist_classification/.venv/Scripts/python.exe mnist_classification/run_experiment.py --config mnist_classification/configs/local.json --plots-only
```

## Dosyalar ve çıktılar

| Dosya | Görev |
|---|---|
| `data.py` | MNIST indirme, MD5, IDX okuma, bölme, train-only epsilon |
| `model.py` | MLP ve güvenli MC Dropout değerlendirme modu |
| `engine.py` | CE/R/FGSM eğitimi, RNG, checkpoint, resume, prediction |
| `metrics.py` | accuracy, error, NLL, normalize Brier |
| `run_experiment.py` | deney yönetimi, source/config imzası, CSV, seed özeti |
| `plots.py` | üç panel, training, örnekler ve olasılıklar |
| `test_correctness.py` | matematik ve resume doğrulaması |
| `configs/*.json` | değişmez smoke / local / full konfigürasyonları |
| `build_notebook.py` | bağımsız Colab notebook üreticisi |
| `verify_artifacts.py` | tamamlanmış deneylerde seed/üye/skor/çıktı tutarlılığı |
| `requirements.txt`, `environment-lock.txt` | taşınabilir gereksinimler ve yerel tam sürüm kaydı |
| `DIFFERENCES.md` | makaleye uyum, farklar, uygulama tercihleri |
| `SONUC_RAPORU.md` | gerçek ölçümler ve sunum/rapor notları |

`outputs/<profil>/`:

- `checkpoints/seed_*/<method>/member_*.pt`: ağırlık + optimizer + RNG (Git dışı).
- `histories/*.csv`, `training_history.csv`: epoch bazında loss ve accuracy; perturbed amaç ve temiz CE ayrı kaydedilir.
- `raw_results.csv`: seed/yöntem/M bazlı tüm gerçek ölçümler; `seed_results/seed_*.csv`: seed dosyaları.
- `summary.csv`: mean/std ve `n_seeds`; `example_predictions.csv`: gerçek örneklerin 10 sınıf olasılığı.
- `probabilities/*.npz`: test sırasına göre üye/MC geçiş olasılıkları (Git dışı); tekrar analiz için.
- `figures/`: `figure2a_comparison`, `training_curves`, `mnist_examples`, `prediction_probabilities`, `high_confidence`, `low_confidence`; hepsi 300 dpi PNG ve vektör PDF.
- `manifest.json`, `config.json`, `data_manifest.json`: kapsam, durum, GPU, paketler, hash'ler, veri sayıları, epsilon bilgisi.
- `splits.npz`, `source_snapshot/`: bölme ve eğitilen kodun kopyası (Git dışı).

Geliştirici doğrulaması için `requirements-dev.txt` kurulduktan sonra `python mnist_classification/validate_notebook.py` notebook'un tüm hücrelerini yerel Jupyter çekirdeğinde smoke profiliyle yürütür. Çıktı notebook'u `outputs/notebook_smoke_executed.ipynb` altında tutulur. Colab bulutunda çalıştırma ile yerel notebook doğrulaması farklıdır.

Veri, sanal ortam, model ağırlıkları ve büyük olasılık dosyaları `.gitignore` kapsamındadır. CSV, rapor, grafik ve notebook paylaşılabilir. Klasör şu anda bir Git deposu olmayabilir; hiçbir otomatik push yapılmaz. Git deposu oluşturmak ve dersin istediği kendi public GitHub deposuna kodu yüklemek kullanıcıya aittir.

## Ders raporu ve sunum

Dersin özgün LaTeX şablonu bu klasörde bulunmuyor; şablon Teams'ten alınmalı ve rapor PDF olarak teslim edilmeli. Sunum 10 dakika (sorular dahil), yaklaşık 8–10 slayt; odak yöntem özetinden çok bu yeniden uygulama, gerçek bulgular ve farklar olmalı. Figure 2(a) bir kalibrasyon diyagramı değildir: proper scoring rule iyileşmesi belirsizlik tahmin kalitesine kanıt sağlar, tek başına tüm anlamlarıyla kalibrasyonu veya OOD güvenilirliğini kanıtlamaz. Bu çalışma NotMNIST/OOD, SVHN veya ImageNet deneylerini kapsamaz.
