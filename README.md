# SGU26K2_MayHoc
## Quy tac chon mo hinh cuoi

1. Chon 4 den 5 ung vien theo Macro-F1 tren CV 5-fold cua tap train.
2. So tung ung vien voi ung vien dung dau bang kiem dinh cap theo fold.
3. Neu khac biet khong co y nghia thong ke thi uu tien theo thu tu:
   Recall Wipro_bulb, do on dinh giua cac fold, don gian va nhanh.
4. Khoa mo hinh. Sau do chi chay tap test mot lan duy nhat.
## Mo hinh cuoi da khoa

Mo hinh: HistGradientBoostingClassifier, class_weight balanced, cac tham so con lai mac dinh.
Co so: CV lap 3x5 tren tap train, Macro-F1 0.9661 +/- 0.0143.
Cac ung vien khong khac nhau co y nghia thong ke, chon theo quy tac tieu chi phu da ghi truoc.
Du phong: xgb_default, Macro-F1 0.9644 +/- 0.0175.
Sau khi khoa, tap test chi duoc chay mot lan duy nhat.