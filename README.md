# 概要
ROSトピックで受講した画像からSSDで物体検出

アルゴリズムはこれ  
https://github.com/weiliu89/caffe/tree/ssd  
利用した実装はこれ  
https://github.com/rykov8/ssd_keras  
利用する重み付けデータはこれ  
https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA

# 起動準備
* レポジトリのダウンロード
````
cd ~/catkin_ws/src
git clone https://github.com/raucha/ssd_ros
````
* ライブラリのインストール
  * `sudo -H pip install keras tensorflow`
* ダウンロードした`weights_SSD300.hdf5`を適当な場所におき，`model.load_weights(....)`を編集してパスを通す
* `img_path`を編集して適当な画像にパスを通す(一度`prediction`を実行しないと以後正常に動かなくなるため．kerasかtensorflowのバグ？)

# 起動
`rosrun pythontest ssd_ros.py`
`
カメラはこれを使う  
http://wiki.ros.org/uvc_camera  

# 発行トピック
topic: /class_num  
入力： カメラ画像

# 購読トピック
topic: /usb_cam/image_raw  
出力：ラベル，検出された四角形の座標，四角形の辺の長さ  
[ラベル, x_min, y_min, x_length, y_length]
