# 📚 DANH MỤC TÀI LIỆU THAM KHẢO KHOA HỌC (ACADEMIC REFERENCES)

## HỆ THỐNG NHẬN DIỆN CẢM XÚC & ĐÁNH GIÁ ĐỘ TẬP TRUNG (EduFocus AI)

> **Quy chuẩn trích dẫn:** Chuẩn học thuật quốc tế **APA 7th Edition**, phân nhóm chuyên đề phục vụ viết Thuyết minh đề tài, Khóa luận tốt nghiệp, Báo cáo Đồ án chuyên ngành và Công bố bài báo khoa học.  
> 🔗 **100% tài liệu đều có link DOI, arXiv hoặc PDF mở (Open Access)** để người đọc có thể click truy cập và đọc trực tiếp toàn văn.

---

## 📌 MỤC LỤC

1. [Mục 1: Kiến Trúc Nền Tảng & Mô Hình Đa Thức Thị Giác - Ngôn Ngữ (Vision-Language Foundation Models)](#mục-1-kiến-trúc-nền-tảng--mô-hình-đa-thức-thị-giác---ngôn-ngữ-vision-language-foundation-models)
2. [Mục 2: Phát Hiện, Căn Chỉnh Khuôn Mặt & Điểm Mốc Giải Phẫu (Face Detection, Alignment & Facial Mesh)](#mục-2-phát-hiện-căn-chỉnh-khuôn-mặt--điểm-mốc-giải-phẫu-face-detection-alignment--facial-mesh)
3. [Mục 3: Cơ Sở Khoa Học Của 25 Đặc Trưng Hình Học Sinh Trắc Học (Facial Geometric Biometrics)](#mục-3-cơ-sở-khoa-học-của-25-đặc-trưng-hình-học-sinh-trắc-học-facial-geometric-biometrics)
4. [Mục 4: Tối Ưu Hóa Trích Chọn Đặc Trưng & Trí Tuệ Bầy Đàn (Feature Selection & Swarm Intelligence)](#mục-4-tối-ưu-hóa-trích-chọn-đặc-trưng--trí-tuệ-bầy-đàn-feature-selection--swarm-intelligence)
5. [Mục 5: Kiến Trúc Phân Loại Hợp Nhất Đa Dạng (Heterogeneous Ensemble & Deep Learning)](#mục-5-kiến-trúc-phân-loại-hợp-nhất-đa-dạng-heterogeneous-ensemble--deep-learning)
6. [Mục 6: Động Học Thời Gian, Lọc Nhiễu Phát Ngôn & Định Danh Học Sinh (Temporal Dynamics & Tracking)](#mục-6-động-học-thời-gian-lọc-nhiễu-phát-ngôn--định-danh-học-sinh-temporal-dynamics--tracking)
7. [Mục 7: Tâm Lý Học Sư Phạm, Đánh Giá Độ Tập Trung & Thang Đo Likert (Educational Psychology & Engagement)](#mục-7-tâm-lý-học-sư-phạm-đánh-giá-độ-tập-trung--thang-đo-likert-educational-psychology--engagement)
8. [Mục 8: Bộ Dữ Liệu Thực Nghiệm Chuẩn Quốc Tế (Benchmark Datasets)](#mục-8-bộ-dữ-liệu-thực-nghiệm-chuẩn-quốc-tế-benchmark-datasets)

---

## Mục 1: Kiến Trúc Nền Tảng & Mô Hình Đa Thức Thị Giác - Ngôn Ngữ (Vision-Language Foundation Models)

1. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021).**  
   _Learning Transferable Visual Models From Natural Language Supervision._  
   In _Proceedings of the 38th International Conference on Machine Learning (ICML 2021)_, PMLR 139, pp. 8748–8763.  
   🔗 Link đọc: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)  
   👉 **Ứng dụng trong EduFocus AI:** Nền tảng backbone CLIP ViT-B/32, cung cấp vector đặc trưng thị giác toàn cục 512D (`f_v`) và bộ mã hóa văn bản Text Encoder cho 7 câu Prompt cảm xúc.

2. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021).**  
   _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale._  
   In _International Conference on Learning Representations (ICLR 2021)_.  
   🔗 Link đọc: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)  
   👉 **Ứng dụng trong EduFocus AI:** Cơ sở lý thuyết của Vision Transformer (ViT), cơ chế chia patch ảnh $32 \times 32$ và Multi-Head Self-Attention trong trích xuất đặc trưng thị giác.

3. **Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. (2021).**  
   _Do Vision Transformers See Like Convolutional Neural Networks?_  
   In _Advances in Neural Information Processing Systems (NeurIPS 2021)_, Vol. 34, pp. 12116–12128.  
   🔗 Link đọc: [arXiv:2108.08810](https://arxiv.org/abs/2108.08810)  
   👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học chứng minh Vision Transformer thiên về nắm bắt ngữ cảnh toàn cục (Global Context) nhưng làm mất các chi tiết hình học cục bộ tần số cao (High-Frequency Local Details), dẫn dắt lý do bắt buộc phải bổ sung nhánh hình học MediaPipe.

4. **Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., & Zou, J. Y. (2022).**  
   _Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning._  
   In _Advances in Neural Information Processing Systems (NeurIPS 2022)_, Vol. 35, pp. 17612–17625.  
   🔗 Link đọc: [arXiv:2203.02053](https://arxiv.org/abs/2203.02053)  
   👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học về hiện tượng "Modality Gap" trong CLIP, chứng minh việc cộng trực tiếp vector thị giác và vector văn bản ($f_z = f_v + f_t$) của CLIVP-FER sơ khai là sai lầm; biện minh cho giải pháp thay thế bằng phép chiếu góc Cosine Similarity 7D.

---

## Mục 2: Phát Hiện, Căn Chỉnh Khuôn Mặt & Điểm Mốc Giải Phẫu (Face Detection, Alignment & Facial Mesh)

5. **Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020).**  
   _RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild._  
   In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020)_, pp. 5203–5212.  
   🔗 Link đọc: [arXiv:1905.00641](https://arxiv.org/abs/1905.00641)  
   👉 **Ứng dụng trong EduFocus AI:** Mô hình phát hiện khuôn mặt SOTA chính của hệ thống, sử dụng kiến trúc Feature Pyramid Network (FPN) và SSH context module để phát hiện 100% học sinh trong lớp, bất chấp góc khuất, khoảng cách xa và ánh sáng phức tạp.

6. **Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Wang, C.-L., Kong, L., Leung, B., & Grundmann, M. (2019).**  
   _MediaPipe: A Framework for Building Perception Pipelines._  
   _arXiv preprint arXiv:1906.08172_.  
   🔗 Link đọc: [arXiv:1906.08172](https://arxiv.org/abs/1906.08172)  
   👉 **Ứng dụng trong EduFocus AI:** Framework phát hiện và trích xuất 478 điểm mốc 3D (3D Facial Landmarks) trên khuôn mặt với độ trễ thấp ở chế độ thời gian thực.

7. **Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019).**  
   _Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs._  
   In _CVPR Workshop on Computer Vision for Augmented and Virtual Reality (CVPRW 2019)_.  
   🔗 Link đọc: [arXiv:1907.06724](https://arxiv.org/abs/1907.06724) | [Google AI Blog](https://research.google/blog/on-device-real-time-face-mesh-with-mediapipe/)  
   👉 **Ứng dụng trong EduFocus AI:** Bài báo chính thức của nhóm nghiên cứu Google công bố mô hình MediaPipe Face Mesh, ước lượng dạng lưới bề mặt khuôn mặt 468/478 điểm tọa độ 3D chuẩn hóa.

---

## Mục 3: Cơ Sở Khoa Học Của 25 Đặc Trưng Hình Học Sinh Trắc Học (Facial Geometric Biometrics)

8. **Ekman, P. (1992).**  
   _An Argument for Basic Emotions._  
   _Cognition & Emotion_, 6(3–4), pp. 169–200.  
   🔗 Link đọc: [DOI: 10.1080/02699939208411068](https://doi.org/10.1080/02699939208411068)  
   👉 **Ứng dụng trong EduFocus AI:** Nền tảng phân loại 7 biểu cảm khuôn mặt cốt lõi (Neutral, Happy, Surprise, Sad, Fear, Angry, Disgust) làm chuẩn nhãn phân loại của hệ thống.

9. **Soukupová, T., & Čech, J. (2016).**  
   _Real-Time Eye Blink Detection using Facial Landmarks._  
   In _Proceedings of the 21st Computer Vision Winter Workshop (CVWW 2016)_, pp. 1–8.  
   🔗 Link đọc: [CVWW 2016 Open Access PDF](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)  
   👉 **Ứng dụng trong EduFocus AI:** Cơ sở toán học của tỷ lệ khung hình mắt $\text{EAR}$ (Chỉ số #8, #9, #13) để đo độ mở mắt, phát hiện chớp mắt và trạng thái sụp mí do mất tập trung/buồn ngủ.

10. **Tian, Y., Kanade, T., & Cohn, J. F. (2001).**  
    _Recognizing Action Units for Facial Expression Analysis._  
    _IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)_, 23(2), pp. 97–115.  
    🔗 Link đọc: [DOI: 10.1109/34.908967](https://doi.org/10.1109/34.908967)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng thực nghiệm chứng minh độ dịch chuyển thẳng đứng của mí trên và mí dưới là chỉ dấu sinh trắc học có độ nhạy cao nhất vùng mắt.

11. **Pantic, M., & Rothkrantz, L. J. (2000).**  
    _Automatic Analysis of Facial Expressions: The State of the Art._  
    _IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)_, 22(12), pp. 1424–1445.  
    🔗 Link đọc: [DOI: 10.1109/34.895972](https://doi.org/10.1109/34.895972)  
    👉 **Ứng dụng trong EduFocus AI:** Xác lập khoảng cách giữa hai đầu lông mày $D_{\text{inter\_brow}}$ (Chỉ số #14) và sự co rút của cơ cau mày trong trạng thái giận dữ (_Angry_).

12. **Ghayoumi, M., & Bansal, A. K. (2016).**  
    _Unifying Geometric Features and Facial Action Units for Improved Performance of Facial Expression Analysis._  
    _arXiv preprint arXiv:1606.00822_.  
    🔗 Link đọc: [arXiv:1606.00822](https://arxiv.org/abs/1606.00822)  
    👉 **Ứng dụng trong EduFocus AI:** Phương pháp kết hợp khoảng cách các điểm mốc hình học (Landmark Distances) và góc mở khóe môi ($\theta_{\text{mouth}}$, Chỉ số #2, #3) với các đơn vị hành động cơ mặt để phân biệt rõ nụ cười (_Happy_) và nét buồn (_Sad_).

13. **Valstar, M. F., & Pantic, M. (2012).**  
    _Fully Automatic Recognition of the Temporal Phases of Facial Actions._  
    _IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)_, 42(1), pp. 28–43.  
    🔗 Link đọc: [DOI: 10.1109/TSMCB.2011.2163401](https://doi.org/10.1109/TSMCB.2011.2163401)  
    👉 **Ứng dụng trong EduFocus AI:** Kỹ thuật chuẩn hóa bất biến tỷ lệ (Scale-Invariance) bằng cách chia cho chiều rộng mặt ($W_{\text{face}}$); phân tích biến dạng rãnh mũi má trong cảm xúc ghê tởm (_Disgust_).

14. **Kazemi, V., & Sullivan, J. (2014).**  
    _One Millisecond Face Alignment with an Ensemble of Regression Trees._  
    In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2014)_, pp. 1867–1874.  
    🔗 Link đọc: [CVPR Open Access PDF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Kazemi_One_Millisecond_Face_2014_CVPR_paper.pdf)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở toán học định vị và căn chỉnh các điểm mốc hình học trên khuôn mặt thời gian thực với độ trễ tính bằng mili-giây.

---

## Mục 4: Tối Ưu Hóa Trích Chọn Đặc Trưng & Trí Tuệ Bầy Đàn (Feature Selection & Swarm Intelligence)

15. **Kennedy, J., & Eberhart, R. (1995).**  
    _Particle Swarm Optimization._  
    In _Proceedings of ICNN'95 - International Conference on Neural Networks_, IEEE, Vol. 4, pp. 1942–1948.  
    🔗 Link đọc: [DOI: 10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968)  
    👉 **Ứng dụng trong EduFocus AI:** Nền tảng của thuật toán bầy đàn PSO cổ điển, cơ chế cập nhật vận tốc và vị trí hạt tìm kiếm giải pháp tối ưu.

16. **Miranda, V., & Fonseca, N. (2002).**  
    _EPSO - Evolutionary Particle Swarm Optimization, a New Algorithm with Applications in Power Systems._  
    In _Proceedings of the IEEE/PES Transmission and Distribution Conference and Exhibition 2002: Asia Pacific_, Yokohama, Japan, Vol. 2, pp. 745–750.  
    🔗 Link đọc: [IEEE PES TDC 2002](https://ieeexplore.ieee.org/abstract/document/1177555)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở của thuật toán EPSO (Evolutionary PSO), kết hợp cơ chế đột biến di truyền ngẫu nhiên để giải bài toán tối ưu tổ hợp chọn lọc đặc trưng nhị phân.

17. **Yao, X., Liu, Y., & Lin, G. (1999).**  
    _Evolutionary Programming Made Faster._  
    _IEEE Transactions on Evolutionary Computation_, 3(2), pp. 82–102.  
    🔗 Link đọc: [DOI: 10.1109/4235.771163](https://doi.org/10.1109/4235.771163)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở toán học của đột biến Cauchy (Cauchy Mutation) với đặc tính đuôi dày (heavy-tailed), tạo bước nhảy xa giúp bầy hạt thoát khỏi bẫy cực trị địa phương trong không gian $\{0, 1\}^{544}$.

18. **Guyon, I., & Elisseeff, A. (2003).**  
    _An Introduction to Variable and Feature Selection._  
    _Journal of Machine Learning Research (JMLR)_, 3, pp. 1157–1182.  
    🔗 Link đọc: [JMLR Open Access PDF](https://www.jmlr.org/papers/v3/guyon03a.html)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở lý thuyết về giảm chiều dữ liệu, hàm thích nghi phạt độ thưa $\ell_0$ (Sparsity Penalty / Occam's Razor) để tinh giản vector từ 544D xuống 289D nhạy nhất.

---

## Mục 5: Kiến Trúc Phân Loại Hợp Nhất Đa Dạng (Heterogeneous Ensemble & Deep Learning)

19. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**  
    _Deep Residual Learning for Image Recognition._  
    In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2016)_, pp. 770–778.  
    🔗 Link đọc: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ chế Skip-Connections ($y = f(x) + x$) áp dụng trong mô hình `EmotionResMLP` để giải quyết triệt để vấn đề suy thoái biểu diễn và triệt tiêu gradient (Vanishing Gradient).

20. **Chen, T., & Guestrin, C. (2016).**  
    _XGBoost: A Scalable Tree Boosting System._  
    In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016)_, pp. 785–794.  
    🔗 Link đọc: [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)  
    👉 **Ứng dụng trong EduFocus AI:** Mô hình phân loại cây quyết định tăng cường gradient `XGBoost Classifier`, tối ưu hàm mất mát qua xấp xỉ Taylor bậc 2 và phạt co cụm lá để bắt trọn các ngưỡng cắt cứng trên 11 đặc trưng hình học.

21. **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).**  
    _Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?_  
    In _Advances in Neural Information Processing Systems (NeurIPS 2022) Datasets and Benchmarks Track_.  
    🔗 Link đọc: [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học quyết định việc kết hợp ResMLP và XGBoost: Mạng nơ-ron sâu chuyên trị không gian nhúng liên tục dày đặc (272D CLIP), trong khi Cây quyết định chuyên trị các ngưỡng cắt trực giao của dữ liệu tỷ lệ dạng bảng (11D hình học).

22. **Shwartz-Ziv, R., & Armon, A. (2022).**  
    _Tabular Data: Deep Learning is Not All You Need._  
    _Information Fusion_, 81, pp. 84–90.  
    🔗 Link đọc: [arXiv:2106.03253](https://arxiv.org/abs/2106.03253)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học chứng minh việc kết hợp Ensemble giữa Mạng Nơ-ron sâu và XGBoost luôn đạt điểm số vượt trội hơn bất kỳ mô hình đơn lẻ nào trên không gian đặc trưng lai (Heterogeneous Representations).

23. **Krogh, A., & Vedelsby, J. (1995).**  
    _Neural Network Ensembles, Cross Validation, and Active Learning._  
    In _Advances in Neural Information Processing Systems (NeurIPS 1995)_, Vol. 7, pp. 231–238.  
    🔗 Link đọc: [NeurIPS Open Access PDF](https://proceedings.neurips.cc/paper_files/paper/1994/file/d709f38ef758b5066ef31b18039b8ce5-Paper.pdf)  
    👉 **Ứng dụng trong EduFocus AI:** Định lý phân rã sai số Ensemble $E = \overline{E} - \overline{A}$; chứng minh rằng kết hợp các mô hình có cấu trúc học khác nhau làm tăng độ đa dạng $\overline{A}$, từ đó triệt tiêu phương sai lỗi hệ thống.

24. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).**  
    _On Calibration of Modern Neural Networks._  
    In _Proceedings of the 34th International Conference on Machine Learning (ICML 2017)_, PMLR 70, pp. 1321–1330.  
    🔗 Link đọc: [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học về hiện tượng "dự đoán quá tự tin" (Overconfidence) của mạng sâu và XGBoost; biện minh cho việc giữ lại mô hình `Baseline EmotionMLP` (10%) làm mỏ neo điều hòa entropy để làm mượt phân phối xác suất.

25. **Wolpert, D. H., & Macready, W. G. (1997).**  
    _No Free Lunch Theorems for Optimization._  
    _IEEE Transactions on Evolutionary Computation_, 1(1), pp. 67–82.  
    🔗 Link đọc: [DOI: 10.1109/4235.585893](https://doi.org/10.1109/4235.585893)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở triết học máy học: Không có một giải thuật duy nhất nào tối ưu trên mọi bài toán; do đó cần phối hợp đa mô hình để bổ khuyết nhược điểm cho nhau.

---

## Mục 6: Động Học Thời Gian, Lọc Nhiễu Phát Ngôn & Định Danh Học Sinh (Temporal Dynamics & Tracking)

26. **Wojke, N., Bewley, A., & Paulus, D. (2017).**  
    _Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)._  
    In _2017 IEEE International Conference on Image Processing (ICIP 2017)_, pp. 3645–3649.  
    🔗 Link đọc: [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở của thuật toán Face Re-ID Tracker, kết hợp chỉ số giao nhau $IoU$, khoảng cách tâm vị trí và độ tương đồng Cosine của vector diện mạo 512D CLIP để bảo toàn 1 Track ID duy nhất cho mỗi học sinh.

27. **Cowie, R., Douglas-Cowie, E., Tsapatsoulis, N., Votsis, G., Kollias, S., Fellenz, W., & Taylor, J. G. (2001).**  
    _Emotion Recognition in Human-Computer Interaction._  
    _IEEE Signal Processing Magazine_, 18(1), pp. 32–80.  
    🔗 Link đọc: [DOI: 10.1109/79.911197](https://doi.org/10.1109/79.911197)  
    👉 **Ứng dụng trong EduFocus AI:** Phân tách chuyển động cơ mặt do giao tiếp lời nói (Speech-related facial actions) và biểu cảm cảm xúc; cơ sở cho bộ lọc Upper-Face Gating để tránh gán nhãn Ngạc nhiên giả khi học sinh đang nói chuyện.

---

## Mục 7: Tâm Lý Học Sư Phạm, Đánh Giá Độ Tập Trung & Thang Đo Likert (Educational Psychology & Engagement)

### 7.1. Công Thức Gốc: Chỉ Số Độ Tập Trung Từ Phân Tích Cảm Xúc Khuôn Mặt

28. **Sharma, P., Esengönül, M., Khanal, S. R., Khanal, T. T., Filipe, V., & Reis, M. J. C. S. (2019).** ⭐ **[TÀI LIỆU GỐC CHO CÔNG THỨC CHỈ SỐ TẬP TRUNG]**  
    _Student Concentration Evaluation Index in an E-learning Context Using Facial Emotion Analysis._  
    In Tsitouridou, M., Diniz, J. A., & Mikropoulos, T. A. (Eds.), _Technology and Innovation in Learning, Teaching and Education (TECH-EDU 2018)_, Communications in Computer and Information Science (CCIS), Vol. 993, pp. 529–538. Springer, Cham.  
    🔗 Link đọc: [DOI: 10.1007/978-3-030-20954-4_40](https://doi.org/10.1007/978-3-030-20954-4_40)  
    👉 **Ứng dụng trong EduFocus AI:** Paper gốc đề xuất công thức **Concentration Index (CI)**: $\text{CI} = \text{DEP} \times \text{EW}$.

    | Cảm xúc      | Trọng số EW (Sharma et al., 2019) | Trọng số EduFocus AI ($w_i$) | Trọng số chuẩn hóa quy đổi ($\tilde{w}_i$) | Ý nghĩa sư phạm & Đối chiếu                                    |
    | :----------- | :-------------------------------: | :--------------------------: | :----------------------------------------: | :------------------------------------------------------------- |
    | **Neutral**  |               0.90                |          **+0.30**           |                  **1.00**                  | Tập trung cao độ, nghe giảng sâu (cao nhất cả 2 hệ thống)      |
    | **Surprise** |               0.60                |          **+0.15**           |                  **0.70**                  | Tò mò, hứng thú trước kiến thức mới                            |
    | **Happy**    |               0.60                |          **+0.10**           |                  **0.60**                  | **Khớp hoàn toàn** với Sharma (hứng thú học tập vừa phải)      |
    | **Sad**      |               0.30                |          **−0.05**           |                  **0.30**                  | **Khớp hoàn toàn** với Sharma (mệt mỏi, chưa hiểu bài)         |
    | **Disgust**  |               0.20                |          **−0.05**           |                  **0.30**                  | Chán nản, không hào hứng với bài học                           |
    | **Angry**    |               0.25                |          **−0.10**           |                  **0.20**                  | Căng thẳng, bực bội, ức chế nhận thức                          |
    | **Fear**     |               0.30                |          **−0.20**           |                  **0.00**                  | Phạt nặng nhất: Lo âu, hoang mang làm tê liệt trí nhớ làm việc |

    **Cách EduFocus AI kế thừa và phát triển từ công thức Sharma:**
    1. **Mở rộng từ đơn cảm xúc sang đa cảm xúc:** Thay vì chỉ dùng 1 cảm xúc trội nhất ($\text{DEP} \times \text{EW}$), EduFocus AI tính **tổng trọng số trên cả 7 kênh xác suất Softmax đồng thời**: $S = \sum_{k=1}^{7} w_k \cdot P_k$.
    2. **Phân cực âm/dương theo tâm lý học hành vi (Fredricks et al., 2004):** Gán trọng số dương cho cảm xúc tích cực và trọng số âm cho cảm xúc tiêu cực, giúp phạt trực tiếp các biểu hiện cản trở tiếp thu.
    3. **Chuẩn hóa Min-Max về thang đo phần trăm trực quan $[0, 100\%]$:** $\text{Score} = \frac{S - w_{\min}}{w_{\max} - w_{\min}} \times 100\%$.

29. **Savchenko, A. V., Savchenko, L. V., & Makarov, I. (2022).** ⭐ **[TÀI LIỆU BỔ TRỢ CHO PHƯƠNG PHÁP TỔNG HỢP ENGAGEMENT TỪ VIDEO]**  
    _Classifying Emotions and Engagement in Online Learning Based on a Single Facial Expression Recognition Neural Network._  
    _IEEE Transactions on Affective Computing_, 13(4), pp. 2132–2143.  
    🔗 Link đọc: [arXiv:2104.04506](https://arxiv.org/abs/2104.04506) | [DOI: 10.1109/TAFFC.2022.3188390](https://doi.org/10.1109/TAFFC.2022.3188390)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng khoa học từ tạp chí IEEE hàng đầu về Affective Computing, chứng minh tính khả thi và hiệu quả của phương pháp **tổng hợp xác suất cảm xúc theo thời gian (Temporal Probability Aggregation)** từ video để đánh giá mức độ tập trung nhóm học sinh.

---

### 7.2. Khung Lý Thuyết Tâm Lý Học Sư Phạm & Phân Nhóm Cảm Xúc

30. **Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004).**  
    _School Engagement: Potential of the Concept, State of the Evidence._  
    _Review of Educational Research_, 74(1), pp. 59–109.  
    🔗 Link đọc: [DOI: 10.3102/00346543074001059](https://doi.org/10.3102/00346543074001059)  
    👉 **Ứng dụng trong EduFocus AI:** Mô hình lý thuyết nền tảng phân loại 3 chiều gắn kết học tập (Behavioral, Emotional, Cognitive Engagement); cơ sở tâm lý học cho việc phân nhóm cảm xúc tích cực (`Neutral, Happy, Surprise` → trọng số dương) và tiêu cực (`Sad, Fear, Angry, Disgust` → trọng số âm).

31. **Whitehill, J., Serpell, Z., Lin, Y.-C., Foster, A., & Movellan, J. R. (2014).**  
    _The Faces of Engagement: Automatic Recognition of Student Engagement from Facial Expressions._  
    _IEEE Transactions on Affective Computing_, 5(1), pp. 86–98.  
    🔗 Link đọc: [DOI: 10.1109/TAFFC.2014.2314605](https://doi.org/10.1109/TAFFC.2014.2314605)  
    👉 **Ứng dụng trong EduFocus AI:** Nghiên cứu thực nghiệm quy mô lớn (34 học sinh, 2.673 mẫu) chứng minh biểu cảm khuôn mặt là chỉ báo đáng tin cậy cho mức độ tập trung; chỉ ra học sinh tập trung sâu duy trì nét mặt bình thản (_Neutral_) trong phần lớn thời lượng.

32. **Pekrun, R. (2006).**  
    _The Control-Value Theory of Achievement Emotions: Assumptions, Corollaries, and Implications for Educational Research and Practice._  
    _Educational Psychology Review_, 18(4), pp. 315–341.  
    🔗 Link đọc: [DOI: 10.1007/s10648-006-9029-9](https://doi.org/10.1007/s10648-006-9029-9)  
    👉 **Ứng dụng trong EduFocus AI:** Thuyết Kiểm soát - Giá trị về Cảm xúc Thành tựu: Chứng minh Lo âu (_Fear_) và Bực bội (_Angry_) làm cạn kiệt tài nguyên Trí nhớ làm việc (Working Memory Depletion), biện minh cho trọng số phạt âm trong công thức CI.

33. **Bosch, N., & D'Mello, S. (2017).**  
    _The Affective Experience of Novice Computer Programmers._  
    _International Journal of Artificial Intelligence in Education_, 27(1), pp. 181–206.  
    🔗 Link đọc: [DOI: 10.1007/s40593-016-0130-9](https://doi.org/10.1007/s40593-016-0130-9)  
    👉 **Ứng dụng trong EduFocus AI:** Mô hình chuyển dịch cảm xúc trong quá trình học tập: Trạng thái tập trung sâu (Flow State), ngạc nhiên nhận thức (Cognitive Disequilibrium) và các dấu hiệu bế tắc/chán nản.

34. **Monkaresi, H., Bosch, N., Calvo, R. A., & D'Mello, S. K. (2017).**  
    _Automated Detection of Engagement Using Video-Based Estimation of Facial Expressions and Heart Rate._  
    _IEEE Transactions on Affective Computing_, 8(1), pp. 27–38.  
    🔗 Link đọc: [DOI: 10.1109/TAFFC.2016.2515087](https://doi.org/10.1109/TAFFC.2016.2515087)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng thực nghiệm chứng minh biểu cảm khuôn mặt từ video là tín hiệu đầu vào hiệu quả để phát hiện engagement tự động trong môi trường học tập.

35. **Grafsgaard, J. F., Wiggins, J. B., Boyer, K. E., Wiebe, E. N., & Lester, J. C. (2013).**  
    _Automatically Recognizing Facial Indicators of Frustration, Confusion, and Interest During Computer-Based Learning._  
    In _Proceedings of the 6th International Conference on Educational Data Mining (EDM 2013)_, pp. 248–255.  
    🔗 Link đọc: [EDM 2013 Open Access PDF](https://educationaldatamining.org/EDM2013/papers/rn_paper_30.pdf)  
    👉 **Ứng dụng trong EduFocus AI:** Bằng chứng thực nghiệm chứng minh các chỉ báo biểu cảm khuôn mặt dự đoán trạng thái bế tắc (Frustration), bối rối (Confusion) và hứng thú (Interest) của học sinh.

---

### 7.3. Cơ Sở Thang Đo & Đánh Giá Định Tính

36. **Joshi, A., Kale, S., Chandel, S., & Pal, D. K. (2015).**  
    _Likert Scale: Explored and Explained._  
    _British Journal of Applied Science & Technology_, 7(4), pp. 396–403.  
    🔗 Link đọc: [DOI: 10.9734/BJAST/2015/14975](https://doi.org/10.9734/BJAST/2015/14975)  
    👉 **Ứng dụng trong EduFocus AI:** Cơ sở phương pháp luận thiết lập thang đo 5 bậc phân khoảng đều (Equal Interval Method) đánh giá mức độ tập trung: Xuất sắc ($80–100\%$), Tốt ($60–80\%$), Trung bình ($40–60\%$), Yếu ($20–40\%$), Rất yếu ($0–20\%$).

37. **Wang, Z., Bergin, C., & Bergin, D. A. (2014).**  
    _Measuring Engagement in Fourth to Twelfth Grade Classrooms: The Classroom Engagement Inventory._  
    _School Psychology Quarterly_, 29(4), pp. 517–535.  
    🔗 Link đọc: [DOI: 10.1037/a0035985](https://doi.org/10.1037/a0035985)  
    👉 **Ứng dụng trong EduFocus AI:** Khung tiêu chuẩn đánh giá mức độ gắn kết lớp học và hệ thống khuyến nghị can thiệp sư phạm tự động theo từng mức xếp loại.

---

## Mục 8: Bộ Dữ Liệu Thực Nghiệm Chuẩn Quốc Tế (Benchmark Datasets)

38. **Li, S., Deng, W., & Du, J. (2017).**  
    _Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Analysis in the Wild._  
    In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017)_, pp. 2584–2593.  
    🔗 Link đọc: [IEEE Xplore CVPR 2017 (DOI: 10.1109/CVPR.2017.277)](https://doi.org/10.1109/CVPR.2017.277) | [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf)  
    👉 **Ứng dụng trong EduFocus AI:** Bộ dữ liệu cảm xúc tự nhiên lớn nhất (15.339 ảnh khuôn mặt) dùng để huấn luyện và kiểm chứng mô hình trong điều kiện đa dạng về góc mặt, độ tuổi và ánh sáng.

39. **Jeong, M., & Ko, B. C. (2018).**  
    _Driver's Facial Expression Recognition in Real-Time for Safe Driving._  
    _Sensors_, 18(12), Article 4270.  
    🔗 Link đọc: [DOI: 10.3390/s18124270](https://doi.org/10.3390/s18124270) | [KMU-FED Lab Page](https://cvpr.kmu.ac.kr/KMU-FED.htm)  
    👉 **Ứng dụng trong EduFocus AI:** Bài báo chính thức công bố bộ dữ liệu KMU-FED của Keimyung University (1.106 ảnh, 12 đối tượng), dùng làm tập dữ liệu kiểm thử độc lập (Person-Independent Benchmark) trong điều kiện thiếu sáng và góc nghiêng.
