# CMP4509 - MNIST gerçek deney sonuçları

Bu rapor yalnızca tamamlanmış ve yeniden denetlenmiş `local` deneyinin dosyalarından üretildi. Çıktı yolu: `outputs/local/`. Makaledeki sayılar veri dosyalarına kopyalanmadı.

## Tamamlanan çalışma

- Deney seed'leri: 42, 123, 2026; M = 1, 2, 3, 4, 5; 10 epoch.
- 48 ayrı model, 63 seed/yöntem/M sonuç satırı, 48 doğrulanmış checkpoint.
- Eğitim 55,000, doğrulama 5,000, test 10,000 örnek. MNIST resmi checksum'ları doğrulandı; eğitim/doğrulama kesişimi boş.
- Donanım: NVIDIA GeForce RTX 4070 Ti SUPER, PyTorch 2.11.0+cu128, CUDA 12.8, Python 3.14.3.
- Kaydedilen eğitim epoch sürelerinin toplamı: 33.07 dakika. Bu süre paket/veri indirme, grafik üretme ve diğer doğrulamaları içermez.
- `toy_regression` değiştirilmeden bağımsız `mnist_classification` uygulaması oluşturuldu.

## Gerçek ölçümler

Hücreler **mean ± sample standard deviation (ddof=1)**, üç bağımsız run seed üzerindendir. Accuracy ve error bu tabloda yüzde, NLL doğal logaritmayla nats, Brier 1/K=1/10 normalizasyonludur. CSV dosyalarında accuracy/error [0,1] oranıdır. Bütün skorlar temiz MNIST test örneklerindendir; test zamanında FGSM saldırısına karşı robustness ölçülmedi.

| Yöntem | M | Accuracy (%) | Error (%) | NLL | Brier (1/K) |
|---|---:|---:|---:|---:|---:|
| Single Model | 1 | 97.577 ± 0.125 | 2.423 ± 0.125 | 0.10360 ± 0.01398 | 0.003885 ± 0.000246 |
| Deep Ensemble | 1 | 97.577 ± 0.125 | 2.423 ± 0.125 | 0.10360 ± 0.01398 | 0.003885 ± 0.000246 |
| Deep Ensemble | 2 | 98.247 ± 0.110 | 1.753 ± 0.110 | 0.06757 ± 0.00360 | 0.002877 ± 0.000145 |
| Deep Ensemble | 3 | 98.397 ± 0.127 | 1.603 ± 0.127 | 0.05952 ± 0.00364 | 0.002627 ± 0.000114 |
| Deep Ensemble | 4 | 98.500 ± 0.075 | 1.500 ± 0.075 | 0.05569 ± 0.00263 | 0.002480 ± 0.000094 |
| Deep Ensemble | 5 | 98.567 ± 0.068 | 1.433 ± 0.068 | 0.05441 ± 0.00150 | 0.002420 ± 0.000101 |
| Ensemble + R | 1 | 97.330 ± 0.285 | 2.670 ± 0.285 | 0.11405 ± 0.01424 | 0.004252 ± 0.000477 |
| Ensemble + R | 2 | 98.130 ± 0.245 | 1.870 ± 0.245 | 0.07228 ± 0.00816 | 0.003111 ± 0.000297 |
| Ensemble + R | 3 | 98.317 ± 0.112 | 1.683 ± 0.112 | 0.06350 ± 0.00520 | 0.002778 ± 0.000201 |
| Ensemble + R | 4 | 98.483 ± 0.130 | 1.517 ± 0.130 | 0.05810 ± 0.00400 | 0.002608 ± 0.000171 |
| Ensemble + R | 5 | 98.507 ± 0.154 | 1.493 ± 0.154 | 0.05584 ± 0.00447 | 0.002504 ± 0.000186 |
| Ensemble + AT | 1 | 97.733 ± 0.127 | 2.267 ± 0.127 | 0.08523 ± 0.00213 | 0.003529 ± 0.000161 |
| Ensemble + AT | 2 | 98.313 ± 0.180 | 1.687 ± 0.180 | 0.05863 ± 0.00877 | 0.002653 ± 0.000327 |
| Ensemble + AT | 3 | 98.433 ± 0.100 | 1.567 ± 0.100 | 0.05554 ± 0.00604 | 0.002501 ± 0.000234 |
| Ensemble + AT | 4 | 98.583 ± 0.060 | 1.417 ± 0.060 | 0.05180 ± 0.00393 | 0.002333 ± 0.000146 |
| Ensemble + AT | 5 | 98.640 ± 0.062 | 1.360 ± 0.062 | 0.05462 ± 0.00764 | 0.002394 ± 0.000272 |
| MC Dropout | 1 | 96.533 ± 0.574 | 3.467 ± 0.574 | 0.13076 ± 0.02456 | 0.005291 ± 0.000910 |
| MC Dropout | 2 | 97.050 ± 0.433 | 2.950 ± 0.433 | 0.11082 ± 0.01916 | 0.004638 ± 0.000724 |
| MC Dropout | 3 | 97.160 ± 0.436 | 2.840 ± 0.436 | 0.10323 ± 0.01763 | 0.004388 ± 0.000658 |
| MC Dropout | 4 | 97.223 ± 0.401 | 2.777 ± 0.401 | 0.10065 ± 0.01790 | 0.004271 ± 0.000617 |
| MC Dropout | 5 | 97.260 ± 0.416 | 2.740 ± 0.416 | 0.09949 ± 0.01865 | 0.004219 ± 0.000652 |

NLL sayısal tabanı (1e-12) denetimi: 60 seed/yöntem/M konfigürasyonunun 6 tanesinde en az bir örnek tabana takıldı; tek konfigürasyonda etkilenen en yüksek örnek sayısı 2. Ayrıntılı minimum olasılıklar ve sayılar `verification.json` içinde.

Seed bazında ham ölçümler `raw_results.csv` ve `seed_results/seed_*.csv` içinde. Bütün satırlarda n_seeds=3; ensemble üyeleri üç tekrar yerine sayılmadı. Single Model satırı clean ensemble'ın ilk üyesiyle aynıdır.

## Makalenin iddiasıyla karşılaştırma

Makale Figure 2(a), M arttığında ensemble yöntemlerinin iyileştiğini ve AT'nin random perturbation ile MC Dropout'tan daha iyi skorlar verdiğini raporlar. Özgün ham MNIST sonuçları bulunmadığından sayısal eşitlik testi yapılmadı; aşağıda bizim sabit kurulumumuz karşılaştırılır.

- Deep Ensemble, M=1 -> 5: error 2.423% -> 1.433%; NLL 0.10360 -> 0.05441; Brier 0.003885 -> 0.002420.
- Ensemble + R, M=1 -> 5: error 2.670% -> 1.493%; NLL 0.11405 -> 0.05584; Brier 0.004252 -> 0.002504.
- Ensemble + AT, M=1 -> 5: error 2.267% -> 1.360%; NLL 0.08523 -> 0.05462; Brier 0.003529 -> 0.002394.
- MC Dropout, M=1 -> 5: error 3.467% -> 2.740%; NLL 0.13076 -> 0.09949; Brier 0.005291 -> 0.004219.
- M=5, AT vs Deep Ensemble: AT ortalaması daha düşük olan metrikler: error, Brier; eşit veya daha yüksek olanlar: NLL.
- M=5, AT vs Ensemble + R: AT ortalaması daha düşük olan metrikler: error, NLL, Brier; eşit veya daha yüksek olanlar: yok.
- M=5, AT vs MC Dropout: AT ortalaması daha düşük olan metrikler: error, NLL, Brier; eşit veya daha yüksek olanlar: yok.

Bu karşılaştırma yalnız gözlenen ortalamalar içindir; n=3 ile evrensel üstünlük veya istatistiksel anlamlılık iddia edilmez. Sonuçları iyileştirmek için test üzerinde lr/epsilon/epoch seçilmedi. Ayrı bir takip deneyi tasarlanırsa hiperparametre kararı validation üzerinden alınmalıdır.

Özgün eğrilerden sapmanın olası nedenleri 10 epoch'luk sabit bütçe, 60k yerine 55k eğitim örneği, clipping ve belirtilmemiş tarihsel Torch ayrıntılarıdır. Bunlar olası açıklamalardır; etkilerini birbirinden ayıran bir ablation çalışması yapılmadığı için nedensel sonuç çıkarılamaz.

## Matematiksel yöntemler ve mimari

- Ağ: 784 -> 200 -> 200 -> 200 -> 10; üç gizli katmanda Linear -> BatchNorm -> ReLU. 240.610 eğitilebilir parametre; MC modelinde her ReLU sonrası p=0.1 dropout (parametre eklemez).
- CE = -log softmax(f(x))[y]. Logits doğrudan `cross_entropy`'ye verilir; önceden softmax uygulanmaz.
- epsilon_j = 0.01*(max_train x_j - min_train x_j). Gerçek maksimum epsilon 0.01000; 69 pikselde epsilon=0.
- FGSM, kaybın birinci mertebe yaklaşımında L-infinity kısıtlı en kötü yönü seçer: delta_j=epsilon_j*sign(dCE/dx_j). R ise bağımsız ±1 yönleri örnekler. İkisinde de [0,1] clip ve CE_clean + CE_perturbed kullanılır.
- Ensemble p_bar = mean_m softmax(logits_m); MC Dropout aynı ifadede m yerine stokastik maskeler kullanır. BatchNorm testte güncellenmez.
- Accuracy=mean(argmax(p_bar)==y); error=1-accuracy; NLL=-mean(log(p_bar[y])); Brier=mean_n mean_k (p_bar_nk-one_hot_nk)^2. NLL tabanı 1e-12.
- -log fonksiyonunun konveksliği karışım NLL'sinin üyelerin ortalama NLL'sini aşmamasını sağlar (sayısal taban hariç); bu, her yeni üye eklendiğinde NLL'nin monoton düşeceğini veya en iyi tek üyeden mutlaka iyi olacağını söylemez. Brier için de konvekslik benzer ortalama-üye karşılaştırması verir; accuracy için böyle bir garanti yoktur.

## Eğitim parametreleri ve farklar

Adam lr=0.1, batch=100, sabit 10 epoch; betas=(0.9,0.999), Adam eps=1e-8, weight decay=0. Scheduler, early stopping ve test tabanlı model seçimi yok. Tüm yöntemlerde son epoch. FP32, deterministic algoritmalar, TF32 ve AMP kapalı. Checkpoint her epoch optimizer ve RNG ile kaydedilir.

Makaleye göre açık tercihler: 55k/5k validation bölmesi, 10 epoch, BN yerleşimi/varsayılanları, [0,1] clipping, PyTorch başlangıcı ve Adam ayrıntıları, eşleştirilmiş tedavi başlangıçları, üç seed ve sample SD error bar'ları. Tam liste `DIFFERENCES.md` içinde. Makaledeki Adam/lr=0.1/batch=100 bilgileri korunmuştur.

## Oluşturulan dosyalar ve çıktılar

`data.py`: veri/MD5/bölme; `model.py`: MLP/MC modu; `engine.py`: eğitim/FGSM/R/checkpoint; `metrics.py`: skorlar; `run_experiment.py`: deney ve CSV; `plots.py`: altı grafik takımı; `test_correctness.py`: matematik/BN/resume; `verify_artifacts.py`: gerçek kayıt denetimi; `build_notebook.py`, `validate_notebook.py`: Colab kaynak paketi ve yerel notebook doğrulaması; `configs/*.json`: profiller; `README.md`, `DIFFERENCES.md`, requirements ve gitignore: çalıştırma/belgeleme. `generate_report.py` bu raporu yalnız tamamlanmış sonuçlardan üretir.

Tüm yöntemlerin checkpoint'leri ve epoch loss/accuracy geçmişleri saklandı. Ham/seed/mean/std CSV'leri, MNIST örnekleri, confidence örnekleri, 10 sınıf olasılıkları, eğitim grafikleri ve Figure 2(a) benzeri üç panel kaydedildi. Her altı grafik yüksek çözünürlüklü PNG ve vektör PDF'tir.

## Doğrulama ve hatalar

1. İlk ortam CPU-only PyTorch içeriyordu; yeni `.venv` içinde CUDA 12.8 PyTorch kuruldu. Eski toy sanal ortamının silinmiş Python 3.11 yoluna işaret etmesi bu klasöre dokunmadan çözüldü.
2. İlk paket kurulumları kullanıcı dizini erişim kısıtına takıldı; izinli kurulumla ayrı proje ortamı oluşturuldu. Paket/veri indirmeleri gerçek olarak tamamlandı.
3. PDF metni yazdırılırken Windows cp1252 kodlaması UnicodeEncodeError verdi; metinler UTF-8 dosyadan okundu. Kaynak PDF sayfaları görsel olarak da incelendi.
4. Yerel notebook doğrulamasında Jupyter çekirdeği kapanış uyarısı görüldü; doğrulama betiğine açık shutdown eklendi. Notebook'un yedi kod hücresi smoke modunda yürütüldü.
5. Beş doğruluk testi geçti: analitik Brier/NLL, softmax ortalaması farkı, MC dropout stochastic/BN sabitliği, FGSM/R yönü ve parametre güncellemesi, dört yöntem için bit düzeyinde CPU ve yerel CUDA checkpoint/resume eşitliği ve imza uyuşmazlığı koruması.
6. Kaydedilmiş tüm test olasılıklarından metrikler yeniden hesaplandı; raw CSV ve sample SD özetleriyle eşleşti. Veri checksum'ları, split ayrıklığı, üye seed'leri ve checkpoint epoch/imzaları doğrulandı. Ayrıntı `verification.json` içinde.

## Tamamlanmayan işler ve kapsam sınırı

- M=10, 15 bu yerel hesaplama bütçesinde çalıştırılmadı. Geniş full.json profili 138 model gerektirir; local profil 48 modelle sınırlandı. Çalıştırılmayan kapsam için sonuç üretilmedi. full.json Colab/GPU'da çalıştırılmaya hazır.
- Colab'ın gerçek ücretsiz bulut oturumunda ana deney çalıştırılmadı; notebook yerel GPU/Jupyter ortamında smoke modunda doğrulandı. Colab donanımıyla aynı sayılar garanti edilmez.
- GitHub'a push yapılmadı. Başlangıç klasörü Git deposu değildi; public repo oluşturma/yükleme bu çalışmada yapılmadı. Ağırlık/veri/sanal ortam ignore kapsamındadır.
- Dersin Teams üzerindeki zorunlu LaTeX şablonu mevcut değildi. Bu Markdown sonuç dökümü dersin zorunlu şablonuyla hazırlanmış nihai PDF raporun yerine geçmez.
- NotMNIST/OOD, SVHN, ImageNet, hiperparametre araması ve geniş seed analizi bu görevin kapsamı dışında.

## Sunum ve raporda anlatılacak noktalar

1. Araştırma sorusu: aynı MLP ile bağımsız ensemble ve adversarial training, accuracy yanında olasılıksal skorları iyileştiriyor mu? Deneyin hedefi Figure 2(a), en iyi MNIST rekoru değil.
2. CE, FGSM'nin birinci mertebe türetimi, probability averaging ve 1/K Brier denklemlerini her ekip üyesi açıklayabilmeli. Logits ortalaması ile olasılık ortalamasını karıştırmayın.
3. MC Dropout'ta BN'yi niçin eval modunda tuttuğunuzu; M ağ ile M maskenin eğitim maliyetinin aynı olmadığını belirtin. Mean±SD üç run seed üzerindedir, üyeler üzerinde değildir.
4. Ana tablo ve üç paneli kullanın; M=1 ile en büyük tamamlanan M'yi karşılaştırın. AT beklentiyi karşılamadıysa sonucu açıkça gösterin; test üzerinde ayar yapmayın.
5. 55k eğitim, 10 epoch, clipping ve tamamlanmayan M değerlerini makaleden farklar slaydında belirtin. Yüksek/düşük confidence örneklerini hata analizi için kullanın; bunlar OOD testi değildir.
6. NLL/Brier proper scoring rule'dur; accuracy aynı kalsa da farklı güven dağılımlarını ayırabilir. Bu skorların iyileşmesi tek başına kusursuz kalibrasyonun kanıtı değildir.
7. Ders sunumu sorular dahil 10 dakika; yaklaşık 8–10 slayt. Sonuç, farklar, sınırlılıklar ve yeniden üretilebilirlik için yeterli süre ayırın; zorunlu LaTeX şablonuna tabloları ve PDF grafikleri ekleyin.
