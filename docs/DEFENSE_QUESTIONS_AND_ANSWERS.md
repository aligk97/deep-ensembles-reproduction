# Savunma soruları ve yanıtları

Bu belge prova içindir. Yanıtları ezberlemek yerine her grup üyesi bütün projeyi kod ve kanıt dosyaları üzerinden anlatabilmelidir. İngilizce teknik terimler anlamlarıyla birlikte korunmuştur. Kişilere verilen sunum bölümleri gerçek akademik katkı beyanı değildir.

## 1. Deep Ensemble nedir?

Aynı görevi öğrenen, parametreleri ayrı tutulan M sinir ağının predictive distribution'larını birleştiririz. Classification'da softmax olasılıkları eşit ağırlıkla ortalanır; regression'da Gaussian üyelerin karışımı oluşur. Tek bir ağın M kopyası değildir: üyeler bağımsız eğitilir. Bizde M=5 ve üç bağımsız run seed vardır.

## 2. Neden random initialization kullanılır?

Nonconvex optimizasyonda farklı başlangıçlar farklı çözümlere ve tahminlere yol açabilir. Data shuffling de farklılık sağlar. Bu diversity (çeşitlilik), tek modelde görülmeyen disagreement'ı gösterebilir. Farklı seed kullanmak üyelerin mutlaka yeterince farklı veya doğru olacağını garanti etmez; toy sol OOD'de ortak hata kalmıştır.

## 3. Neden bagging kullanılmadı?

Makale ağların tüm eğitim verisini görmesini tercih eder. N kez replacement ile bootstrap örneklemede benzersiz örnek oranı yaklaşık 1−exp(−1)=0.632'dir; küçük veride bu kayıp üyelerin gücünü azaltabilir. Ağlarda initialization/shuffling zaten randomization sağlar. Biz yeni bagging karşılaştırması yapmadık; tercih makalenin seçilen protokolünü izler.

## 4. Proper scoring rule nedir?

Gerçek koşullu dağılım q iken beklenen skoru doğru dağılımı bildirerek optimize ettiğimiz kuraldır. Loss yönünde, log loss beklentisi H(q)+KL(q||p) olur; KL negatif olamadığı için p=q optimumdur. Strict propriety uygun koşullarda eşitliğin yalnız doğru dağılımda olmasıdır. Bu population özelliği finite-data modelin veya OOD tahminlerinin otomatik calibrated olduğu anlamına gelmez.

## 5. NLL neden kullanılır?

Negative log-likelihood yalnız doğru/yanlış sınıfı değil, gözlenen hedefe verilen olasılığı değerlendirir. Yanlış ama aşırı emin classification tahmini yüksek NLL alır. Gaussian regression'da hem mean hem variance öğrenilebilir. Herhangi bir yüksek NLL sonucu silinmez; bizim toy tablolarında çok büyük NLL'ler vardır.

## 6. MSE neden tek başına belirsizlik ölçmez?

MSE bir conditional mean tahminini eğitir. Sabit varyanslı Gaussian varsayılırsa NLL ile ölçek farkı dışında eşdeğerdir, fakat varyansı öğrenmez. MSE ensemble üyelerinin tahmin varyansı model disagreement'ını gösterebilir, observation noise'u açıkça modellemez. Bu nedenle MSE proxy NLL'sini öğrenilmiş Gaussian likelihood ile aynı yöntem gibi anlatmıyoruz.

## 7. Gaussian NLL nasıl türetilir?

p(y|x)=(2πv)^(-1/2)exp(−(y−μ)^2/(2v)) ile başlayın. −log alınca 0.5log(2π)+0.5log(v)+(y−μ)^2/(2v) elde edilir. İlk terim parametrelere göre sabittir; training'de atılabilir. Log variance geniş tahminleri, residual/variance terimi dar ama yanlış tahminleri cezalandırır. Board'da density'den başlayarak türetin.

## 8. Softplus neden kullanılır?

Variance pozitif olmalıdır; raw ağ çıkışı negatif olabilir. Softplus(a)=log(1+exp(a)) pozitif, smooth bir dönüşümdür; türevi sigmoid'dir. PyTorch'un numerically stable uygulaması kullanılır ve 1e−6 eklenir. Bu floor normalized y koordinatlarındadır; fiziksel varyansa dönüştürülürken y_scale² ile çarpılır.

## 9. Ensemble variance nasıl türetilir?

Üye indeksi Z uniform olsun. E[Y]=E_Z μ_Z=μ_bar ve E[Y²]=E_Z(v_Z+μ_Z²). Var(Y)=E[Y²]−E[Y]² açılımı, mean(v_m)+mean((μ_m−μ_bar)²) verir. Bu law of total variance'dır. Burada divisor M kullanılır; üç run seed'inin sample SD'sindeki M−1 türü düzeltme ile karıştırmayın.

## 10. Aleatoric ve epistemic uncertainty farkı nedir?

Aleatoric uncertainty veri/hedefin koşullu değişkenliğidir; epistemic uncertainty model/parametre bilgisinin yetersizliğine bağlıdır. Gaussian ensemble'da average predicted variance ve between-member mean variance bunlar için model-dependent göstergelerdir. Bunları gerçek noise ve posterior variance'ın kusursuz ayrıştırılması gibi sunmayın. Bütün üyeler aynı yanlış extrapolation yapabilir.

## 11. Adversarial training neden faydalı olabilir?

Gözlenen girdinin küçük çevresinde yüksek-loss yönüne de doğru hedefle eğitim uygulayarak predictive distribution'ı yerel olarak smooth yapabilir. Makale bunu uncertainty için inceler; bizim MNIST sonuçları seçilen protokolde olumlu yöndedir. Bu, her veri setinde veya her OOD türünde fayda garantisi değildir. AT'nin bütün M değerlerinde istatistiksel olarak üstün olduğunu iddia etmiyoruz.

## 12. FGSM nasıl çalışır?

Taylor yaklaşımında loss(x+δ)≈loss(x)+δᵀg. Her koordinat için |δ_i|≤ε_i altında maksimum δ_i=ε_i sign(g_i) ile alınır. MNIST'te pertürbasyon [0,1]'e clip edilir; ε_i gerçek training split'indeki pixel range'in yüzde biridir. Clean ve perturbed loss toplanır; 0.5 ile ortalanmaz.

## 13. Random perturbation ile FGSM farkı nedir?

+R aynı büyüklükte bağımsız random sign yönü seçer, gradient'e bakmaz. +AT input gradient sign ile loss'u yerel olarak artıran yönü hedefler. Aynı ε büyüklüğü aynı training etkisi demek değildir. Paired initialization/shuffling seed'leri yöntem karşılaştırmasını düzenler; iki yöntem ayrı ağırlıklarla eğitilir.

## 14. MC Dropout nasıl çalışır?

Her seed için dropout ile eğitilen tek model vardır. Evaluation'da üç Dropout layer aktif kalır, M farklı maskeyle forward yapılır ve probabilities ortalanır. M, dropout için eğitilmiş ağ sayısı değil stochastic pass sayısıdır. Beşli ensemble'ın beş bağımsız ağırlık setinden farklı bir maliyet/çeşitlilik yapısına sahiptir.

## 15. BatchNorm neden evaluation mode'da tutulur?

Train mode'da batch statistics ve running buffers değişebilir; bu, dropout maskesinin etkisini başka bir değişkenle karıştırır. Model eval yapılır, yalnız Dropout layer'ları train durumuna alınır. Running mean, running variance ve num_batches_tracked değişmediği smoke ve full inference kayıtlarında kontrol edilir. Modelin tamamına train() demek doğru uygulama değildir.

## 16. Neden logits değil olasılıklar ortalanır?

Ensemble predictive distribution, üye dağılımlarının mixture'ıdır. Softmax nonlinear olduğundan softmax(mean logits) ile mean(softmax logits) eşit değildir. Makalenin sınıflandırma mixture tanımı ikincisidir. Girdi probability tensor'u [M,N,K], M boyunca ortalama sonrası [N,K]'dır.

## 17. Brier Score nedir?

Bir örnek için (1/K)Σ_k(p_k−one_hot(y)_k)²; veri üzerinde ayrıca N ile ortalanır. K=10 olduğundan sınıflar üzerindeki mean, paper'ın 1/K normalizasyonunu sağlar. Normalize etmeyen bir implementation'ın sayısı bizimkinden on kat büyük olabilir. Düşük değer daha iyidir; Brier ve accuracy farklı özellikleri ölçer.

## 18. Predictive entropy nedir?

Önce M üyenin probabilities ortalamasını alın, sonra H=−Σ p_bar log(p_bar) hesaplayın. Natural log birimi nat'tir ve 0log0=0 güvenli ele alınır. Ortalama member entropy aynı şey değildir: iki farklı sınıfa kesin tahmin veren üyenin kendi entropy'si sıfır, birleşik entropy log2 olur. Predictive entropy yalnız epistemic uncertainty ölçmez.

## 19. Maksimum entropy neden ln(10)?

Uniform u_k=1/K için KL(p||u)=lnK−H(p)≥0 yazın. Buradan H≤lnK çıkar ve uniform dağılımda eşitlik olur. One-hot dağılımda H=0'dır. K=10 için üst sınır yaklaşık 2.302585 nat; iki eşit sınıf için ln2 yaklaşık .693147 nat'tir.

## 20. AUROC neyi ölçer?

Rastgele seçilmiş OOD örneğinin entropy skorunun rastgele ID örneğininkinden büyük olma olasılığına, eşitliklerin yarısını ekler. MNIST=0, NotMNIST=1 ve yüksek entropy=OOD olarak kodlanır. Perfect separation 1, aynı skorlar .5 verir. Threshold-independent bir ranking ölçüsüdür; deployment threshold veya her OOD örneği için güvenlik garantisi değildir.

## 21. NotMNIST neden kullanıldı?

Makalenin Section 3.5 / Figure 3(a) deneyini izler. Boyut ve grayscale yapısı MNIST ile uyumludur, fakat harf sınıfları rakam etiket uzayının dışındadır. Bu kontrollü değişim unknown-class uncertainty'yi incelemeye yarar. Tek bir OOD veri setinden bütün distribution shift türlerine genelleme yapmıyoruz.

## 22. OOD örneklerinde neden accuracy hesaplanmadı?

Digit classifier yalnız 0–9 arasında sınıf üretebilir; NotMNIST A–J metadata etiketleri bu uzaya ait değildir. A'yı 0'a, B'yi 1'e eşitleyip accuracy ölçmek yapay ve yanlış bir görev olur. Harflerde entropy, ID/OOD gap ve AUROC değerlendiriyoruz. MNIST accuracy/NLL/Brier ise checkpoint'in kendi Figure 2(a) koşusuyla karşılaştırılıyor.

## 23. Neden M=10 ve M=15 çalıştırılmadı?

Protokol ücretsiz GPU bütçesinde üç seed'i koruyarak M≤5 seçti. Daha büyük M, ensemble yöntemleri için ek bağımsız model eğitimi gerektirir; dropout yalnız ek forward yapabilir ama bu, tüm yöntemlerin M=15 karşılaştırmasını sağlamaz. Orijinal Figure 2(a)'nın daha geniş, Figure 3(a)'nın M=1,5,10 aralığını tam karşılamadığımız bir deviation'dır. Çalıştırılmayan boyutlar için sonuç iddia etmiyoruz.

## 24. Neden üç seed kullanıldı?

Ders şablonu en az üç independent run ister; compute bütçesi de dikkate alınmıştır. 42,123,2026 run'larının ortalaması ve sample SD'si raporlanır. Bir ensemble'ın beş üyesi beş deney tekrarı değildir. Üç seed sınırlı evidence sağlar; confidence interval veya statistical significance testi yerine geçmez.

## 25. Toy Regression neden kısmi başarı?

Yöntemler implement edilip ölçümler alınsa da aranan uncertainty davranışı her bölgede çıkmadı. Sol OOD'de bütün yöntemlerin bütün seed'lerinde noisy-target 95% coverage sıfırdır; bu yerelde ve T4'te sürer. Deep Ensemble'ın ID clean RMSE'si yaklaşık 5.063 iken sol OOD'de 79.728'dir. Başarılı nitel reproduction diye sunmak bu gözlemlerle tutarlı değildir.

## 26. Sol OOD uncertainty problemi neden oluşmuş olabilir?

ReLU ağlarının benzer parçalı doğrusal uzantıları ortak mean hatası oluşturabilir; Gaussian variance head gözlenmeyen bölgede floor'a yaklaşabilir. Üye disagreement'ı shared bias'ı yakalamayabilir. Bunlar hypotheses: sorunu açıklamak için yeni kontrollü ablation veya eğitim yapılmadı. Aynı başarısızlığın iki ortamda bulunması mekanizmayı tek başına kanıtlamaz.

## 27. Colab ve yerel sonuçlar neden tam aynı değil?

Aynı config ve seed, farklı GPU/PyTorch/CUDA sürümünde bitwise training trajectory garantilemez. Yerelde RTX 4070 Ti SUPER / PyTorch 2.11.0+cu128, Colab'da T4 / 2.14.1+cu130 vardır. Floating-point kernels ve stochastic optimization olası katkılardır; bu etkileri izole eden bir deney yapmadık. Bütün ana sonuçları Colab'dan, bütün yerel sonuçları ek karşılaştırmadan almak environment cherry-picking'i önler.

## 28. Makalenin hangi iddialarını destekledik?

Seçilen MNIST ve MNIST/NotMNIST protokolünde ensemble-versus-dropout nitel eğilimini destekledik. M=5 +AT accuracy yaklaşık %98.61, OOD AUROC .9668; dropout karşılıkları %97.32 ve .7779'dur. Toy'da güvenilir sol extrapolation uncertainty'sini yeniden üretemedik. Makalenin tamamını, ImageNet/SVHN'yi veya universal calibration iddiasını doğruladığımızı söylemiyoruz.

## 29. Temel sınırlılıklarımız neler?

Üç seed, tek OOD veri seti, M≤5, bir küçük sabit toy training draw, 10 epoch MNIST ve original-paper farklılıkları vardır. Significance testi ve ek ECE/reliability-diagram çalışması yoktur. NLL/Brier probabilistic quality için faydalıdır ama tam calibration kanıtı değildir. Kaydedilen sürelerin kapsamları farklı olduğundan adil speedup benchmark'ı yapılmaz.

## 30. Confidence interval ile prediction interval ve sample SD farkı nedir?

Prediction interval gelecekteki Y dağılımına yöneliktir; modelimizde μ±zσ kullanılır. Üç seed'in sample SD'si skorların run'lar arası yayılımıdır. Seed mean için confidence interval başka bir istatistiksel nesnedir ve burada hesaplanmadı. Ana toy grafik ±3σ, noisy coverage ±1.96σ, diagnostic clean-curve coverage ise farklı hedefe bakar.

## 31. Exact mixture NLL ile moment-matched NLL neden farklı?

Gauss karışımı genellikle tek Gaussian değildir. Aynı mean/variance'a sahip Gaussian, dar ayrı bileşenler arasına çok farklı density verebilir. Toy'da exact mixture NLL bazen moment approximation'dan çok daha büyüktür. Dosyalardaki nll_exact_mixture ve nll_moment kolonlarını birbirinin yerine kullanmıyoruz; büyük olumsuz değerleri de saklıyoruz.

## 32. AUROC .3790 nasıl yorumlanır?

Yerel Ensemble M=1, seed=123 için entropy çoğu OOD/ID çiftini yanlış yönde sıralamıştır; .5'in altı kötü bir sonuçtur. Skorun yönü önceden belirlenmiştir, bu seed'i iyi göstermek için entropy işaretini değiştirmeyiz. Sonuç raporda ve raw CSV'de korunur. Tek seed başarısızlığı bütün ensemble büyüklükleri veya ortamlar için sonuç değildir.

## 33. Windows/Linux ordering düzeltmesi bilimsel sonucu değiştiriyor mu?

Dosyaların içeriği ve preprocessing değişmedi; sorted(Path) Windows'ta case-insensitive, Linux'ta case-sensitive idi. 18,724 örnekten 16,719 konum farklıydı. Images, labels ve sample IDs birlikte yerel referans sırasına alınır; hem tensor hem order hash eşleşir. Bu, MC Dropout maskeleriyle görüntü eşleşmesini ve doğrulanmış protokolü korur; guard'ı kaldırma veya rastgele subsampling değildir.

## 34. 48 modelin OOD'de gerçekten tekrar kullanıldığını nasıl biliyoruz?

Gerçek MNIST Colab audit'inin 48 SHA-256 değeri arşivdeki checkpoint byte'larıyla eşleşir. OOD manifestinin checkpoint inventory'si aynı değerleri taşır; classifier_reference manifesti ve raw CSV de MNIST export'uyla byte-identical'dır. OOD new_training=false ve real_colab_inference_completed=true kaydı vardır. Son teslim hazırlığı bu kayıtları okudu; checkpoint inference'ını yeniden çalıştırmadı.

## 35. Mean ± sample SD nasıl hesaplandı?

Her method/M veya method/region için üç run skorunu alın. Mean=sum(a_s)/3; sample variance=sum((a_s−mean)²)/(3−1); SD kareköküdür. Ensemble mixture variance'da divisor M kullanılmasının nedeni farklıdır. Son teslim kontrolü saved CSV'den bu aggregation'ı otomatik tekrar hesaplar; aynı seed'in üyelerini ayrı run gibi saymaz.

## 36. Bir ensemble yüksek entropy verdiğinde nedenini tek başına anlayabilir miyiz?

Hayır. Her üye aynı ambiguous dağılımı verebilir veya üyeler farklı kesin sınıflarda anlaşamayabilir. Predictive entropy iki durumda da artabilir. H(mean p)−mean H(p) disagreement için daha özel bir quantity'dir, fakat bu projede yeni ana deney metriği olarak raporlanmadı. OOD'de yüksek entropy gerekli olabilecek bir davranıştır; tek başına doğru epistemic decomposition garantisi değildir.

## 37. Calibration, accuracy ve OOD detection aynı şey mi?

Accuracy argmax'ın doğru olmasını; calibration probability ile uzun dönem frequency uyumunu; OOD detection ise bilinmeyen örnekleri bir score ile ayırmayı sorar. Yüksek accuracy düşük NLL veya iyi OOD AUROC'yu zorunlu kılmaz. Biz NLL/Brier ve entropy tabanlı OOD skorlarını ölçtük; bütün calibration özelliklerini ispatlamadık. Toy kötü extrapolation, classification başarısının yanında açıkça kalır.

## 38. GPU sürelerinden hangi maliyet sonucunu çıkarabiliriz?

Notebook/mode/runtime kapsamları farklıdır: Toy complete workload, MNIST invocation, OOD inference groups ve plotting dahil invocation aynı tür ölçüm değildir. Ensemble ayrı parametre setleri ve eğitim maliyeti taşır; MC Dropout tek ağırlık setini saklayıp M forward yapar. Kaydedilen T4 süreleri bütçeye sığdığını gösterir; yöntemler arasında kontrollü efficiency benchmark'ı değildir.

## 39. Başka hangi işler öğrencilere kaldı?

İsim, öğrenci numarası ve gerçek katkılar doldurulmalı; her üye kodu, türetimleri ve başarısız sonuçları bağımsız doğrulamalıdır. Güncel ders duyurularında sayfa sınırı ve atanan sunum günü kontrol edilmeli. Hazırlanan temiz içeriğin mevcut public repository'ye yayınlanması kullanıcı onayından sonra yapılmalı. Sonra final PDF yeniden derlenip kontrol edilmeli ve resmi teslim yapılmalıdır.

## Board pratiği

Her üye boş kağıtta Gaussian density'den NLL'ye, mixture second moment'ten total variance'a ve uniform dağılıma KL'den entropy üst sınırına geçişi türetsin. Sonra iki farklı one-hot üyenin predictive entropy'sini hesaplasın. Son olarak bir CSV satırından hangi config/seed/checkpoint kaydına gidileceğini göstersin.
