"""Write Turkish tables and findings from paired, verified regional scores."""
from .plot_v4_lr_comparison import REGION_NAMES
from .train_v1_v2_models import CONDITIONS

MAIN = {"rmse_noisy":"RMSE (y)", "rmse_true":"RMSE (x³)", "predictive_nll":"NLL",
        "coverage_95":"%95 kapsama", "coverage_99_73":"±3σ kapsama",
        "interval_width_95":"%95 genişlik", "interval_width_99_73":"±3σ genişlik"}


def fmt(value):
    return f"{value:.2f}".replace(".", ",")


def write_report(metrics, manifest, table, pairs):
    lr = manifest["learning_rate"]
    parts = ["# V4 · Adam öğrenme oranının etkisi", "",
             f"V2-A’da Adam LR 0,1 iken V4’te {fmt(lr)} kullanıldı. Amaç, öğrenme oranının hataya ve belirsizlik kalibrasyonuna etkisini ölçmektir. Bu ayar, makalenin Şekil 1 düzenine eklediğimiz bir deneydir.",
             "", f"V4 koşusu `{manifest['run_id']}`; V2-A kaynağı `{manifest['baseline_run_id']}/arms/v2_a`. Üye başına 3.000 epoch, 12.000 güncelleme, batch 5 ve weight decay 0 kullanıldı. Tek gizli katman 100 ReLU biriminden oluşur. Veri, üye başlangıç ve karıştırma seed’leri, Adam betaları ve epsilonu, varyans tabanı ve FGSM aynı tutuldu. Her ağ aynı seed’in 20 örneğiyle eğitildi.",
             "", "Her hücre seed 42, 123 ve 2026 için ortalama ± örneklem standart sapmasıdır (ddof=1). RMSE ve genişlik y biriminde, kapsama yüzde olarak verilmiştir. NLL, gürültülü gözlemin negatif log yoğunluğudur. Sol ve sağ OOD’de 1.000’er, eğitim aralığında 2.000 test noktası vardır. Birleşik OOD 2.000 dış noktanın tamamından doğrudan hesaplanır. Genel tablo aynı seed’in 4.000 test noktasını birleştirir."]
    for method in CONDITIONS:
        parts += ["", f"## {method.title}", "", "| Bölge | Deney | "+" | ".join(MAIN.values())+" |",
                  "|---|---|"+"---:|"*len(MAIN)]
        sub = table[table.model==method.name].set_index(["region","metric"])
        for region, label in REGION_NAMES.items():
            for prefix, version in (("v2a","V2-A"),("v4","V4")):
                cells=[]
                for metric in MAIN:
                    row = sub.loc[(region,metric)]
                    scale = 100 if metric.startswith("coverage") else 1
                    cells.append(f"{fmt(row[prefix+'_mean']*scale)} ± {fmt(row[prefix+'_std']*scale)}")
                parts.append("| "+" | ".join([label,version,*cells])+" |")
    deep = table[table.model=="nll_ensemble_5"].set_index(["region","metric"])
    ood = deep.loc["combined_ood"]
    rmse = ood.loc["rmse_true"]
    width = ood.loc["interval_width_95"]
    coverage = ood.loc["coverage_95"]
    parts += ["", "## Birleşik OOD’de doğruluk ve kapsama", "",
              f"NLL ensemble için LR değiştiğinde temiz RMSE {fmt(rmse.v2a_mean)} → {fmt(rmse.v4_mean)}, "
              f"%95 kapsama %{fmt(coverage.v2a_mean*100)} → %{fmt(coverage.v4_mean*100)} oldu. "
              f"Genişlik oranı {fmt(width.v4_mean/width.v2a_mean)} kattır."]
    if width.v4_mean > width.v2a_mean and coverage.v4_mean > coverage.v2a_mean and rmse.v4_mean > rmse.v2a_mean:
        parts.append("Kapsamadaki artış çok daha geniş aralıklarla elde edildi. Aynı test noktalarında "
                     "ortalama tahminin hatası da büyüdü. Bu ölçümlerde yüksek LR genel bir iyileşme sağlamadı.")
    parts += ["","## NLL ensemble sonuçları","",
              "Aşağıdaki farklar V4 eksi V2-A’dır. Daha düşük RMSE ve NLL daha iyidir. Kapsama için hedefe yakınlık, genişlikle birlikte okunur. %95 ve ±3σ aralıkları ensemble momentlerinden gelir. ±3σ’nın Gaussian karşılığı yaklaşık %99,73’tür. Deep Ensemble NLL ise beş Gaussian bileşenin karışım yoğunluğundan hesaplanır. CSV’de karışım kuantillerine dayalı kapsama da ayrı etiketlidir. MSE ensemble’ın NLL ve kapsaması, üye varyansına 1e-6 taban uygulayan Gaussian proxy’dir.",
              ""]
    for region, label in REGION_NAMES.items():
        part=deep.loc[region]
        cov=part.loc["coverage_95"]
        err_b,err_v=(abs(cov[k]-.95)*100 for k in ("v2a_mean","v4_mean"))
        verb="yaklaştı" if err_v<err_b else "uzaklaştı" if err_v>err_b else "aynı uzaklıkta kaldı"
        parts.append(f"- **{label}.** Temiz RMSE farkı {fmt(part.loc['rmse_true','paired_mean_delta'])}, karışım NLL farkı {fmt(part.loc['predictive_nll','paired_mean_delta'])}. %95 genişlik farkı {fmt(part.loc['interval_width_95','paired_mean_delta'])}. Ortalama kapsama %95 hedefine {verb} (hedeften sapma {fmt(err_b)} → {fmt(err_v)} yüzde puan).")
    parts += ["", "## Seed ve yöntem duyarlılığı", ""]
    for method in CONDITIONS:
        data=table[(table.model==method.name)&(table.region=="combined_ood")].set_index("metric")
        parts.append(f"- {method.title}, birleşik OOD: V4 eksi V2-A temiz RMSE {fmt(data.loc['rmse_true','paired_mean_delta'])}, NLL {fmt(data.loc['predictive_nll','paired_mean_delta'])}, %95 genişlik {fmt(data.loc['interval_width_95','paired_mean_delta'])}.")
    deep_pairs=pairs[(pairs.model=="nll_ensemble_5") & (pairs.metric=="rmse_true")]
    for region in ("left_ood","right_ood"):
        entries=deep_pairs[deep_pairs.region==region].sort_values("seed")
        parts.append(f"- {REGION_NAMES[region]} temiz RMSE farkı " +
                     ", ".join(f"seed {r.seed} için {fmt(r.delta)}" for r in entries.itertuples())+".")
    for method in [m for m in CONDITIONS if m.gaussian]:
        row=table[(table.model==method.name)&(table.metric=="aleatoric_variance")&(table.region=="in_domain")].iloc[0]
        parts.append(f"- {method.title} için eğitim aralığındaki ortalama aleatorik varyans V2-A’da {fmt(row.v2a_mean)}, V4’te {fmt(row.v4_mean)}. Veri üretimindeki gerçek gürültü varyansı 9’dur.")
    parts += ["", "Sol ve sağ OOD ayrı raporlanır çünkü kübik fonksiyonda iki taraftaki dışa tahmin hataları ve bant genişlikleri farklılaşabilir. Üç seed ile görülen değişim sabit test örneklerine aittir. Örneklem standart sapması seed değişkenliğini ölçer."]
    (metrics/"findings_tr.md").write_text("\n".join(parts)+"\n",encoding="utf-8")

