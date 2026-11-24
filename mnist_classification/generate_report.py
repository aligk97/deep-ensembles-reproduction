"""Write the Turkish report only from a completed, audited experiment."""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = {"single": "Single Model", "ensemble": "Deep Ensemble", "random": "Ensemble + R",
          "adversarial": "Ensemble + AT", "mc_dropout": "MC Dropout"}


def generate(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and verification["status"] == "passed"
    c, env = manifest["config"], manifest["environment"]
    assert not c["train_limit"] and not c["test_limit"], "Smoke results cannot generate the main scientific report"
    data = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    s = pd.read_csv(output / "summary.csv")
    raw = pd.read_csv(output / "raw_results.csv")
    max_m = max(c["ensemble_sizes"])
    lines = ["# CMP4509 - MNIST gerçek deney sonuçları", "",
             f"Bu rapor yalnızca tamamlanmış ve yeniden denetlenmiş `{c['name']}` deneyinin dosyalarından üretildi. "
             f"Çıktı yolu: `outputs/{c['name']}/`. Makaledeki sayılar veri dosyalarına kopyalanmadı.", "",
             "## Tamamlanan çalışma", "",
             f"- Deney seed'leri: {', '.join(map(str, c['seeds']))}; "
             f"M = {', '.join(map(str, c['ensemble_sizes']))}; {c['epochs']} epoch.",
             f"- {manifest['trained_models']} ayrı model, {len(raw)} seed/yöntem/M sonuç satırı, "
             f"{verification['checkpoints']} doğrulanmış checkpoint.",
             f"- Eğitim {data['train_n']:,}, doğrulama {data['validation_n']:,}, test {data['test_n']:,} örnek. "
             f"MNIST resmi checksum'ları doğrulandı; eğitim/doğrulama kesişimi boş.",
             f"- Donanım: {env['gpu'] or 'CPU'}, PyTorch {env['torch']}, CUDA {env['cuda']}, Python {env['python'].split()[0]}.",
             f"- Kaydedilen eğitim epoch sürelerinin toplamı: {manifest['total_training_epoch_seconds']/60:.2f} dakika. "
             "Bu süre paket/veri indirme, grafik üretme ve diğer doğrulamaları içermez.",
             "- `toy_regression` değiştirilmeden bağımsız `mnist_classification` uygulaması oluşturuldu.", "",
             "## Gerçek ölçümler", "",
             "Hücreler **mean ± sample standard deviation (ddof=1)**, üç bağımsız run seed üzerindendir. "
             "Accuracy ve error bu tabloda yüzde, NLL doğal logaritmayla nats, Brier 1/K=1/10 normalizasyonludur. "
             "CSV dosyalarında accuracy/error [0,1] oranıdır. Bütün skorlar temiz MNIST test örneklerindendir; "
             "test zamanında FGSM saldırısına karşı robustness ölçülmedi.", "",
             "| Yöntem | M | Accuracy (%) | Error (%) | NLL | Brier (1/K) |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in ("single", "ensemble", "random", "adversarial", "mc_dropout"):
        for _, row in s[s.method == method].sort_values("M").iterrows():
            def pm(metric, scale=1, digits=5):
                return f"{row[metric+'_mean']*scale:.{digits}f} ± {row[metric+'_std']*scale:.{digits}f}"
            lines.append(f"| {LABELS[method]} | {int(row.M)} | {pm('accuracy',100,3)} | "
                         f"{pm('classification_error',100,3)} | {pm('nll')} | {pm('brier',1,6)} |")
    floors = verification["nll_floor_diagnostics"]
    affected = sum(row["n_clamped"] > 0 for row in floors)
    maximum = max(row["n_clamped"] for row in floors)
    lines.extend(["", f"NLL sayısal tabanı (1e-12) denetimi: {len(floors)} seed/yöntem/M "
                  f"konfigürasyonunun {affected} tanesinde en az bir örnek tabana takıldı; "
                  f"tek konfigürasyonda etkilenen en yüksek örnek sayısı {maximum}. "
                  "Ayrıntılı minimum olasılıklar ve sayılar `verification.json` içinde."])
    lines.extend(["", "Seed bazında ham ölçümler `raw_results.csv` ve `seed_results/seed_*.csv` içinde. "
                  "Bütün satırlarda n_seeds=3; ensemble üyeleri üç tekrar yerine sayılmadı. "
                  "Single Model satırı clean ensemble'ın ilk üyesiyle aynıdır.", "",
                  "## Makalenin iddiasıyla karşılaştırma", "",
                  "Makale Figure 2(a), M arttığında ensemble yöntemlerinin iyileştiğini ve AT'nin random "
                  "perturbation ile MC Dropout'tan daha iyi skorlar verdiğini raporlar. Özgün ham MNIST "
                  "sonuçları bulunmadığından sayısal eşitlik testi yapılmadı; aşağıda bizim sabit kurulumumuz karşılaştırılır.", ""])
    for method in ("ensemble", "random", "adversarial", "mc_dropout"):
        group = s[s.method == method].set_index("M")
        first, last = group.loc[1], group.loc[max_m]
        lines.append(f"- {LABELS[method]}, M=1 -> {max_m}: error "
                     f"{100*first.classification_error_mean:.3f}% -> {100*last.classification_error_mean:.3f}%; "
                     f"NLL {first.nll_mean:.5f} -> {last.nll_mean:.5f}; "
                     f"Brier {first.brier_mean:.6f} -> {last.brier_mean:.6f}.")
    at = s[(s.method == "adversarial") & (s.M == max_m)].iloc[0]
    for other in ("ensemble", "random", "mc_dropout"):
        row = s[(s.method == other) & (s.M == max_m)].iloc[0]
        wins = [label for key, label in (("classification_error", "error"), ("nll", "NLL"), ("brier", "Brier"))
                if at[key+"_mean"] < row[key+"_mean"]]
        not_wins = [label for key, label in (("classification_error", "error"), ("nll", "NLL"), ("brier", "Brier"))
                    if at[key+"_mean"] >= row[key+"_mean"]]
        lines.append(f"- M={max_m}, AT vs {LABELS[other]}: AT ortalaması daha düşük olan metrikler: "
                     f"{', '.join(wins) or 'yok'}; eşit veya daha yüksek olanlar: {', '.join(not_wins) or 'yok'}.")
    lines.extend(["", "Bu karşılaştırma yalnız gözlenen ortalamalar içindir; n=3 ile evrensel üstünlük veya "
                  "istatistiksel anlamlılık iddia edilmez. Sonuçları iyileştirmek için test üzerinde lr/epsilon/epoch "
                  "seçilmedi. Ayrı bir takip deneyi tasarlanırsa hiperparametre kararı validation üzerinden alınmalıdır.", "",
                  "Özgün eğrilerden sapmanın olası nedenleri 10 epoch'luk sabit bütçe, 60k yerine 55k eğitim örneği, "
                  "clipping ve belirtilmemiş tarihsel Torch ayrıntılarıdır. Bunlar olası açıklamalardır; "
                  "etkilerini birbirinden ayıran bir ablation çalışması yapılmadığı için nedensel sonuç çıkarılamaz.", "",
                  "## Matematiksel yöntemler ve mimari", "",
                  "- Ağ: 784 -> 200 -> 200 -> 200 -> 10; üç gizli katmanda Linear -> BatchNorm -> ReLU. "
                  "240.610 eğitilebilir parametre; MC modelinde her ReLU sonrası p=0.1 dropout (parametre eklemez).",
                  "- CE = -log softmax(f(x))[y]. Logits doğrudan `cross_entropy`'ye verilir; önceden softmax uygulanmaz.",
                  "- epsilon_j = 0.01*(max_train x_j - min_train x_j). "
                  f"Gerçek maksimum epsilon {data['epsilon_max']:.5f}; {data['epsilon_zero_pixels']} pikselde epsilon=0.",
                  "- FGSM, kaybın birinci mertebe yaklaşımında L-infinity kısıtlı en kötü yönü seçer: "
                  "delta_j=epsilon_j*sign(dCE/dx_j). R ise bağımsız ±1 yönleri örnekler. "
                  "İkisinde de [0,1] clip ve CE_clean + CE_perturbed kullanılır.",
                  "- Ensemble p_bar = mean_m softmax(logits_m); MC Dropout aynı ifadede m yerine stokastik "
                  "maskeler kullanır. BatchNorm testte güncellenmez.",
                  "- Accuracy=mean(argmax(p_bar)==y); error=1-accuracy; NLL=-mean(log(p_bar[y])); "
                  "Brier=mean_n mean_k (p_bar_nk-one_hot_nk)^2. NLL tabanı 1e-12.",
                  "- -log fonksiyonunun konveksliği karışım NLL'sinin üyelerin ortalama NLL'sini aşmamasını "
                  "sağlar (sayısal taban hariç); bu, her yeni üye eklendiğinde NLL'nin monoton düşeceğini "
                  "veya en iyi tek üyeden mutlaka iyi olacağını söylemez. Brier için de konvekslik benzer "
                  "ortalama-üye karşılaştırması verir; accuracy için böyle bir garanti yoktur.", "",
                  "## Eğitim parametreleri ve farklar", "",
                  f"Adam lr={c['learning_rate']}, batch={c['batch_size']}, sabit {c['epochs']} epoch; "
                  "betas=(0.9,0.999), Adam eps=1e-8, weight decay=0. Scheduler, early stopping ve test "
                  "tabanlı model seçimi yok. Tüm yöntemlerde son epoch. FP32, deterministic algoritmalar, "
                  "TF32 ve AMP kapalı. Checkpoint her epoch optimizer ve RNG ile kaydedilir.", "",
                  "Makaleye göre açık tercihler: 55k/5k validation bölmesi, 10 epoch, BN yerleşimi/varsayılanları, "
                  "[0,1] clipping, PyTorch başlangıcı ve Adam ayrıntıları, eşleştirilmiş tedavi başlangıçları, "
                  "üç seed ve sample SD error bar'ları. Tam liste `DIFFERENCES.md` içinde. "
                  "Makaledeki Adam/lr=0.1/batch=100 bilgileri korunmuştur.", "",
                  "## Oluşturulan dosyalar ve çıktılar", "",
                  "`data.py`: veri/MD5/bölme; `model.py`: MLP/MC modu; `engine.py`: eğitim/FGSM/R/checkpoint; "
                  "`metrics.py`: skorlar; `run_experiment.py`: deney ve CSV; `plots.py`: altı grafik takımı; "
                  "`test_correctness.py`: matematik/BN/resume; `verify_artifacts.py`: gerçek kayıt denetimi; "
                  "`build_notebook.py`, `validate_notebook.py`: Colab kaynak paketi ve yerel notebook doğrulaması; "
                  "`configs/*.json`: profiller; `README.md`, `DIFFERENCES.md`, requirements ve gitignore: çalıştırma/belgeleme. "
                  "`generate_report.py` bu raporu yalnız tamamlanmış sonuçlardan üretir.", "",
                  "Tüm yöntemlerin checkpoint'leri ve epoch loss/accuracy geçmişleri saklandı. Ham/seed/mean/std "
                  "CSV'leri, MNIST örnekleri, confidence örnekleri, 10 sınıf olasılıkları, eğitim grafikleri ve "
                  "Figure 2(a) benzeri üç panel kaydedildi. Her altı grafik yüksek çözünürlüklü PNG ve vektör PDF'tir.", "",
                  "## Doğrulama ve hatalar", "",
                  "1. İlk ortam CPU-only PyTorch içeriyordu; yeni `.venv` içinde CUDA 12.8 PyTorch kuruldu. "
                  "Eski toy sanal ortamının silinmiş Python 3.11 yoluna işaret etmesi bu klasöre dokunmadan çözüldü.",
                  "2. İlk paket kurulumları kullanıcı dizini erişim kısıtına takıldı; izinli kurulumla ayrı proje "
                  "ortamı oluşturuldu. Paket/veri indirmeleri gerçek olarak tamamlandı.",
                  "3. PDF metni yazdırılırken Windows cp1252 kodlaması UnicodeEncodeError verdi; "
                  "metinler UTF-8 dosyadan okundu. Kaynak PDF sayfaları görsel olarak da incelendi.",
                  "4. Yerel notebook doğrulamasında Jupyter çekirdeği kapanış uyarısı görüldü; "
                  "doğrulama betiğine açık shutdown eklendi. Notebook'un yedi kod hücresi smoke modunda yürütüldü.",
                  "5. Beş doğruluk testi geçti: analitik Brier/NLL, softmax ortalaması farkı, "
                  "MC dropout stochastic/BN sabitliği, FGSM/R yönü ve parametre güncellemesi, "
                  "dört yöntem için bit düzeyinde CPU ve yerel CUDA checkpoint/resume eşitliği ve imza uyuşmazlığı koruması.",
                  "6. Kaydedilmiş tüm test olasılıklarından metrikler yeniden hesaplandı; raw CSV ve "
                  "sample SD özetleriyle eşleşti. Veri checksum'ları, split ayrıklığı, üye seed'leri ve "
                  "checkpoint epoch/imzaları doğrulandı. Ayrıntı `verification.json` içinde.", "",
                  "## Tamamlanmayan işler ve kapsam sınırı", ""])
    missing = sorted(set((1, 2, 3, 4, 5, 10, 15)) - set(c["ensemble_sizes"]))
    if missing:
        lines.append(f"- M={', '.join(map(str, missing))} bu yerel hesaplama bütçesinde çalıştırılmadı. "
                     "Geniş full.json profili 138 model gerektirir; local profil 48 modelle sınırlandı. "
                     "Çalıştırılmayan kapsam için sonuç üretilmedi. full.json Colab/GPU'da çalıştırılmaya hazır.")
    lines.extend(["- Colab'ın gerçek ücretsiz bulut oturumunda ana deney çalıştırılmadı; notebook yerel GPU/Jupyter "
                  "ortamında smoke modunda doğrulandı. Colab donanımıyla aynı sayılar garanti edilmez.",
                  "- GitHub'a push yapılmadı. Başlangıç klasörü Git deposu değildi; public repo oluşturma/yükleme "
                  "bu çalışmada yapılmadı. Ağırlık/veri/sanal ortam ignore kapsamındadır.",
                  "- Dersin Teams üzerindeki zorunlu LaTeX şablonu mevcut değildi. Bu Markdown sonuç dökümü "
                  "dersin zorunlu şablonuyla hazırlanmış nihai PDF raporun yerine geçmez.",
                  "- NotMNIST/OOD, SVHN, ImageNet, hiperparametre araması ve geniş seed analizi bu görevin kapsamı dışında.", "",
                  "## Sunum ve raporda anlatılacak noktalar", "",
                  "1. Araştırma sorusu: aynı MLP ile bağımsız ensemble ve adversarial training, accuracy yanında "
                  "olasılıksal skorları iyileştiriyor mu? Deneyin hedefi Figure 2(a), en iyi MNIST rekoru değil.",
                  "2. CE, FGSM'nin birinci mertebe türetimi, probability averaging ve 1/K Brier denklemlerini "
                  "her ekip üyesi açıklayabilmeli. Logits ortalaması ile olasılık ortalamasını karıştırmayın.",
                  "3. MC Dropout'ta BN'yi niçin eval modunda tuttuğunuzu; M ağ ile M maskenin eğitim maliyetinin "
                  "aynı olmadığını belirtin. Mean±SD üç run seed üzerindedir, üyeler üzerinde değildir.",
                  "4. Ana tablo ve üç paneli kullanın; M=1 ile en büyük tamamlanan M'yi karşılaştırın. "
                  "AT beklentiyi karşılamadıysa sonucu açıkça gösterin; test üzerinde ayar yapmayın.",
                  "5. 55k eğitim, 10 epoch, clipping ve tamamlanmayan M değerlerini makaleden farklar slaydında belirtin. "
                  "Yüksek/düşük confidence örneklerini hata analizi için kullanın; bunlar OOD testi değildir.",
                  "6. NLL/Brier proper scoring rule'dur; accuracy aynı kalsa da farklı güven dağılımlarını ayırabilir. "
                  "Bu skorların iyileşmesi tek başına kusursuz kalibrasyonun kanıtı değildir.",
                  "7. Ders sunumu sorular dahil 10 dakika; yaklaşık 8–10 slayt. Sonuç, farklar, sınırlılıklar ve "
                  "yeniden üretilebilirlik için yeterli süre ayırın; zorunlu LaTeX şablonuna tabloları ve PDF grafikleri ekleyin.", ""])
    target = ROOT / "SONUC_RAPORU.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
