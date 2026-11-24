"""Paired comparisons and Turkish findings for the V3 hyperparameter sweep."""
from itertools import combinations

import pandas as pd


def abbreviate(arm):
    return "V3-" + arm.split("_")[1].upper() if arm.startswith("v3_") else arm.upper().replace("_","-")


def paired_comparisons(scores, references):
    data = pd.concat([scores,references], ignore_index=True)
    data["label"] = data.arm_id.map(abbreviate)
    pairs = [("V3-B","V3-H"),("V3-D","V3-I"),("V3-D","V3-F"),("V3-G","V3-M"),("V3-C","V3-K"),
             *combinations(("V3-C","V3-A","V3-J"),2),("V2-A","V3-L"),("V2-A","V2-C"),("V1","V2-A"),
             ("V3-C","V3-N")]
    for arm in scores.arm_id.unique():
        pairs.extend((ref,abbreviate(arm)) for ref in ("V1","V2-A"))
    columns = ["experiment_version","source_run_id","arm_id","seed","model","region","metric","value","batch_size",
               "learning_rate","weight_decay","epochs","optimizer_steps","n_samples"]
    output=[]
    for left,right in dict.fromkeys(pairs):
        a, b = [data.loc[data.label.eq(label),columns] for label in (left,right)]
        keys = ["seed","model","region","metric","n_samples"]
        paired = a.merge(b,on=keys,suffixes=("_left","_right"),validate="one_to_one")
        if paired.empty:
            continue
        paired.insert(0,"comparison",left+" / "+right)
        paired["right_minus_left"] = paired.value_right-paired.value_left
        paired["run_id"] = scores.run_id.iloc[0]
        paired["experiment_version"] = "V3"
        paired["status"] = "verified"
        output.append(paired)
    return pd.concat(output,ignore_index=True)


def report(raw, metrics, mean, comparisons):
    deep = mean[mean.model.eq("nll_ensemble_5")].copy()
    deep["label"] = deep.arm_id.map(abbreviate)
    lines=["# V3 · Toy Regression", "",
           f"Koşu `{raw.name}`. Amaç, aynı üç veri çekilişinde Adam öğrenme oranı, batch boyutu, epoch sayısı ve ağırlık azaltımının tahminlere etkisini ölçmektir.",
           "", "## Deney düzeni", "",
           "Her seed için 20 gözlem ve dört model ailesi kullanıldı. 100 ReLU birimli tek gizli katman, Gaussian varyans başı ve aynı veri üzerinde bağımsız ensemble üyeleri Şekil 1 düzenini izler. Ağlar bootstrap kullanmaz. FGSM adımı gözlenen x aralığının %1’idir. Eğitim kaybı temiz ve adversarial Gaussian NLL toplamıdır. PyTorch başlatması, epoch sayıları ve üç-seed bölgesel testler bu çalışmanın tercihleridir. [Makale](https://arxiv.org/abs/1612.01474)",
           "", "V1 başlangıç koşuludur (LR 0,001 ve batch 5). V2-A öğrenme oranını 0,1’e yükseltir. V2-C aynı öğrenme oranında tek tam batch kullanır. V3-A–M için LR 0,1’dir. V3-N keşif amaçlıdır. Yalnız MSE için LR 0,01 kullanır. C ve N diğer ailelerde aynı ayarlara sahiptir. Aynı seed ve üye indeksindeki tek Gaussian model ile Gaussian ensemble’ın ilk üyesi projedeki başlangıç ve sıra seed’lerini paylaşır. Her ağ kendi Adam durumuyla eğitilir.",
           "", "| V3 | Batch | Epoch | Üye başına Adam adımı | Weight decay |",
           "|---|---:|---:|---:|---:|"]
    for row in deep.drop_duplicates("arm_id").itertuples():
        lines.append(f"| {row.label} | {row.batch_size} | {row.epochs} | {row.optimizer_steps} | {row.weight_decay:g} |")
    lines+=["", "## Sol OOD, eğitim aralığı ve sağ OOD", "",
            "Aşağıdaki değerler NLL ensemble için üç seed ortalamasıdır. RMSE temiz x³ eğrisine uzaklığı ölçer. Kapsama gürültülü gözlemlerin aralıkta kalan oranıdır. Genişlik y birimindedir. %95 bandı ensemble momentlerinden hesaplanır. Gerçek mixture NLL ve eşit kuyruklu mixture kapsaması CSV’de ayrı ölçümlerdir.", "",
            "| V3 | Bölge | RMSE (x³) ± SS | Mixture NLL | %95 kapsama | %95 genişlik |",
            "|---|---|---:|---:|---:|---:|"]
    for (arm,region), sub in deep.groupby(["label","region"]):
        if region=="combined_ood":
            continue
        table = sub.set_index("metric")
        def fmt(key):
            return f"{table.loc[key,'value']:.3f}"
        lines.append(f"| {arm} | {region} | {fmt('rmse_true')} ± {table.loc['rmse_true','sample_std']:.3f} | {fmt('predictive_nll')} | {fmt('coverage_95')} | {fmt('interval_width_95')} |")
    lines+=["","## Sonuçların yorumu",""]
    for region in ("left_ood","in_domain","right_ood"):
        table = deep[(deep.region==region)&(deep.metric=="rmse_true")].set_index("label")
        best,worst,variable = table.value.idxmin(),table.value.idxmax(),table.sample_std.idxmax()
        lines.append(f"- {region} için en düşük ortalama RMSE {best} ({table.loc[best,'value']:.2f}), en yüksek {worst} ({table.loc[worst,'value']:.2f}). Seed’ler arası en büyük RMSE standart sapması {variable} koşulunda ({table.loc[variable,'sample_std']:.2f}).")
    lines+=["","Aşağıda eşleşmiş karşılaştırmalarda ikinci ayar eksi ilk ayarın üç-seed ortalama RMSE farkı yer alır. Pozitif fark ikinci ayarda daha büyük hatayı gösterir. Her iki tarafta aynı yöntem, seed ve test noktaları kullanılır. Tüm aileler ve ölçümler comparisons.csv içindedir.",
            "", "| Karşılaştırma | Sol OOD | Eğitim aralığı | Sağ OOD |", "|---|---:|---:|---:|"]
    subset = comparisons[(comparisons.model=="nll_ensemble_5")&(comparisons.metric=="rmse_true")]
    for pair in ["V3-B / V3-H","V3-D / V3-I","V3-D / V3-F","V3-G / V3-M","V3-C / V3-K","V3-C / V3-A","V3-A / V3-J","V2-A / V3-L","V1 / V2-A","V1 / V3-A","V2-A / V3-A","V3-C / V3-N"]:
        item = subset[subset.comparison==pair].groupby("region").right_minus_left.mean()
        if len(item):
            lines.append(f"| {pair} | {item.left_ood:.2f} | {item.in_domain:.2f} | {item.right_ood:.2f} |")
    v2 = subset[(subset.comparison=="V2-A / V2-C")&(subset.region=="right_ood")]
    if len(v2):
        deltas=", ".join(f"seed {r.seed} için {r.right_minus_left:+.2f}" for r in v2.itertuples())
        lines.append(f"\nV2-C eksi V2-A sağ OOD RMSE farkı {deltas}. Üç seed ortalaması {v2.right_minus_left.mean():+.2f}. Tek seed grafiği yalnız o veri ve başlangıç çekilişini gösterir.")
    mixed = comparisons[(comparisons.comparison=="V3-C / V3-N") &
                        (comparisons.model=="mse_ensemble_5") &
                        (comparisons.metric=="rmse_true")].groupby("region").right_minus_left.mean()
    lines.append(f"\nMSE ailesinde C’den N’ye geçişte LR 0,1’den 0,01’e iner. "
                 f"RMSE farkı sol OOD’de {mixed.left_ood:+.2f}, eğitim aralığında {mixed.in_domain:+.2f}, "
                 f"sağ OOD’de {mixed.right_ood:+.2f}.")
    lines+=["", "C ile K üye başına 10.000 güncelleme yapar. C her adımda tek gözlem, K beş gözlem kullanır. C, A ve J aynı 500 epoch içinde sırasıyla 10.000, 2.000 ve 500 güncelleme yapar.",
            "", "Geniş aralık kapsama oranını yükseltebilir. Bu nedenle kapsama, genişlik, NLL ve RMSE birlikte okunmalıdır. MSE ensemble öğrenilmiş aleatorik varyans üretmez. Onun Gaussian proxy NLL ve kapsaması, üyelerin popülasyon varyansına 1e-6 alt sınır uygulanarak hesaplanır. Tek ağlarda epistemik varyans tahmin edilmez. Combined OOD ölçümleri iki taraftaki tüm gözlemlerden doğrudan hesaplanır.",
            "", "Üç seed için standart sapmada ddof=1 kullanılır. Sıralamalar bu sabit test çekilişlerinin betimsel sonuçlarıdır. OOD etiketleri eğitimde veya ayar seçiminde kullanılmadı. Tarihsel V3 dosyaları bulunamadığı için sonuçlar yeni ölçümlerdir."]
    (metrics/"findings_tr.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

