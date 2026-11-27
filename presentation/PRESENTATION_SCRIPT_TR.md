# Sunum prova metni (Türkçe)

Bu paylaşım öneridir; gerçek proje katkılarını iddia etmez. Anlatım 7:30, soru-cevap 2:30 planlanmıştır. Ders yönergesine göre slayttan veya nottan okumak yasaktır. Metni prova için kullanın; sunumda kendi cümlelerinizle anlatın. İsim/numara alanlarını doldurun ve süreyi sesli prova ile ölçün.

## 1. Deep Ensembles Reproduction

Student 1 — 15 seconds

CMP4509 için 2017 Deep Ensembles makalesinin seçilmiş deneylerini yeniden ürettik. Regresyon, rakam sınıflandırma ve bilinmeyen harf görüntülerini ele aldık. Desteklenen bulguları ve başarısız olan belirsizlik sonucunu ayrı ve açık biçimde sunacağız.

## 2. Predictive uncertainty

Student 1 — 45 seconds

Başlangıç problemimiz, bir sinir ağının alışılmadık bir girdi geldiğinde de cevap vermesidir. Rakam sınıflandırmada yüksek accuracy, yani doğruluk, bir harf üzerindeki olasılık dağılımının güvenilir olduğunu göstermez. Makale, birbirinden bağımsız birkaç ağın tahmin dağılımlarını birleştirmeyi öneriyor. Üç soruya odaklandık: varyans öğrenmek oyuncak regresyonda belirsizliği iyileştiriyor mu, ensemble sınıflandırma skorlarında MC Dropout'tan daha iyi mi ve NotMNIST üzerinde belirsizlik artıyor mu? Makalenin tüm deneylerini değil, bu soruları ücretsiz GPU bütçesinde sınadık.

## 3. Deep Ensembles method

Student 1 — 90 seconds

Her model bir dağılım tahmin ediyor ve proper scoring rule, yani doğru dağılımı beklenen skorda ödüllendiren bir kuralla öğreniyor. Log loss için beklenen kayıp, gerçek entropy ile negatif olmayan KL divergence toplamıdır. Regresyondaki Gaussian NLL, log-varyans cezasını ve varyansa bölünen karesel hatayı içerir. Böylece çok dar ve hatalı tahmin ile gereksiz geniş tahmin arasında denge kurulur. Softplus, varyansı pozitif tutar. Ensemble eşit ağırlıklı bir mixture, yani karışımdır. Ortalaması üye ortalamalarının ortalamasıdır. İkinci momenti açtığımızda toplam varyans, ortalama üye varyansı ile üye ortalamalarının varyansının toplamı olur. Bunlar gözlem belirsizliği ve model anlaşmazlığı için model varsayımlarına bağlı yorumlardır. Sınıflandırmada önce softmax olasılıklarını ortalıyoruz, sonra predictive entropy hesaplıyoruz. İki kesin ama farklı sınıf tahmini yapan üyenin kendi entropy'si sıfırdır, birleşimin entropy'si log ikidir. FGSM, kaybın girdiye göre gradient sign yönünde küçük pertürbasyon ekler ve temiz ile pertürbe kayıpları toplar. MC Dropout ise tek modelden maske örnekler; BatchNorm sabit kalır.

## 4. Reproduction setup

Student 2 — 45 seconds

Tamamlanmış deneylerde kullanılan final config dosyalarını koruduk. Run seed değerleri 42, 123 ve 2026'dır. Toy Regression her koşuda aynı yirmi gözlemi ve üç bin full-batch epoch kullanır. MNIST, her biri iki yüz birimli üç gizli katmanlı MLP ile on epoch eğitilir; validation ayrımı yalnız eğitim verisinden yapılır. Kırk sekiz model, her ensemble yöntemi için seed başına beş üye ve dropout için bir modelden oluşur. OOD bunların aynısını tekrar kullanır. Üç gerçek T4 koşusunu ana sonuç olarak, yerel koşuları ek karşılaştırma olarak raporluyoruz. Standart sapma, ddof bir olan sample standard deviation'dır; beş üye beş bağımsız deney seed'i sayılmaz.

## 5. Toy Regression: partial reproduction

Student 2 — 60 seconds

Toy Regression sonucu kısmi reproduction'dır. Burada en iyi görünen koşuyu değil, önceden belirlenen ilk seed olan 42'nin Deep Ensemble grafiğini gösteriyoruz. Sol tarafta ortalama tahmini gerçek kübik eğriden çok uzaklaşırken belirsizlik bandı yetersiz kalıyor. Dört yöntemin bütün seed'lerinde sol OOD yüzde doksan beş noisy-target coverage değeri sıfırdır. Aynı başarısızlık yerelde de sürüyor. Grafikteki bant makaledeki gibi üç standart sapmadır. Sayısal coverage ise yeni gürültülü gözlemler üzerinde ayrı hesaplanan yaklaşık bir nokta doksan altı standart sapmalık aralıktır. Bu iki kavramı karıştırmıyoruz. Ortak extrapolation bias ve variance collapse olası açıklamalardır; kontrollü bir nedensel ablation yapmadık. Varyans öğrenmek ve ensemble kullanmak veri dışındaki belirsizliği otomatik olarak güvenilir kılmıyor.

## 6. MNIST Classification

Student 2 — 45 seconds

MNIST'te nitel eğilim daha olumlu. Beş üyede Ensemble plus AT accuracy değeri yüzde 98.61, beş dropout geçişinde yüzde 97.32. Bu protokolde AT'nin ortalama NLL ve normalize Brier değerleri de daha düşüktür. Brier hesabı makaledeki bir bölü on normalizasyonunu içerir. Her nokta üç bağımsız eğitim koşusunun ortalamasıdır; error bar sample standard deviation gösterir. Her M artışının her metriği iyileştirdiğini söylemiyoruz. Örneğin AT classification error, dört üyeden beşe geçerken hafif yükseliyor. Üç seed ayrıca istatistiksel anlamlılık iddiası kurmaya yeterli bir significance testi oluşturmaz.

## 7. MNIST + NotMNIST predictive entropy

Student 3 — 70 seconds

OOD grafiğinin üst satırı bilinen on bin MNIST rakamını, alt satırı okunabilen on sekiz bin yedi yüz yirmi dört NotMNIST harfini gösteriyor. Bilinen iki bozuk görüntü çıkarıldı. Burada yeni model eğitimi yoktur. Önceki Colab sınıflandırma koşusunun aynı kırk sekiz modelini kullanıyor ve hash değerlerini doğruluyoruz. Bilinen sınıflarda predictive entropy düşük kalıyor. Harflerde üye sayısı arttıkça ensemble entropy'si dropout'tan daha fazla büyüyor. Bununla birlikte bazı bilinmeyen girdiler düşük entropy almaya devam ediyor; her OOD örneğinin güvenilir tespiti garanti edilmiyor. Harfler, rakam sınıflandırıcısının etiket uzayının dışındadır. Bu nedenle harfler için rakam accuracy'si yerine belirsizlik ve OOD sıralamasını inceliyoruz.

## 8. OOD separation and reproducibility

Student 3 — 50 seconds

Ek AUROC analizimiz entropy skorlarının ID ve OOD örneklerini ne kadar doğru sıraladığını özetliyor. Beş üyede AT yaklaşık 0.9668, sade ensemble 0.9653, dropout 0.7779 değerinde. Tabloda bütün yöntemler, seed yayılımları ve entropy gap birlikte bulunuyor. AUROC ve gap, orijinal histogram deneyine ek analizlerimizdir. Provenance kontrolleri bir işletim sistemi sorununu da ortaya çıkardı: görüntüler aynıydı ama Windows ve Linux dosya adlarını farklı sıralıyordu. Görüntüleri ve etiketleri birlikte referans sırasına almak hash eşleşmesini geri getirdi. Üç seed, tek OOD veri seti, sınırlı M aralığı ve çözülmemiş toy başarısızlığı temel sınırlılıklardır.

## 9. Conclusions and questions

Student 3 — 30 seconds

Sonucumuz sınırlı ama nettir. Ensemble yöntemleri bu protokolde seçilen sınıflandırma ve bilinmeyen sınıf eğilimini destekledi. Toy Regression ise sol extrapolation bölgesinde güvenilir belirsizliği yeniden üretemedi. Olumlu sonuç seçmek yerine her iki bulguyu ve bütün seed'leri koruduk. Repository kaynakları, gerçek çalışma kanıtlarını, türetimleri ve tam tabloları içeriyor. Varsayımları, toplam varyans türetimini ve sınırlılıkları tartışmaya hazırız.
