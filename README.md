# Zed Yolo Ros Node

Bu projede Zed Stereo kamera kullanılarak object detection yapılmaktadır.
https://github.com/stereolabs/zed-yolo adresinde bulunun zed-python-sample kullanılmıştır. Kendi trafik tabelalarımız için eğitiğimiz Yolo modelemizi kullandık. Sonra projeyi sistemimize uygun olması için ROS nodu olcak şekilde düzenlemeler yaptık.

Kamerdan fotoğraflar alınırken https://github.com/stereolabs/zed-ros-wrapper kullanılarak subscribe ile fotoğraflar alınmıştır.
