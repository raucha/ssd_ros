# 概要
ROSトピックで受講した画像からSSDで物体検出

アルゴリズムはこれ  
https://github.com/weiliu89/caffe/tree/ssd  
利用した実装はこれ  
https://github.com/rykov8/ssd_keras
利用する重み付けデータはこれ  
https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA

## 動作方法
* ダウンロードした`weights_SSD300.hdf5`を適当な場所におき，`model.load_weights(....)`を編集してパスを通す
* `img_path`を編集して適当な画像にパスを通す(一度`prediction`を実行しないと以後正常に動かなくなるため．kerasかtensorflowのバグ？)

# 発行トピック
/class_num

# 購読トピック
/usb_cam/image_raw
