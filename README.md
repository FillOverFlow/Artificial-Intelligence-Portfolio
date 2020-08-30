# Artificial-Intelligence-Portfolio
Artificial Intelligence Portfolio

[![Screen-Shot-2563-08-18-at-23-16-59.png](https://i.postimg.cc/C5gM4G5y/Screen-Shot-2563-08-18-at-23-16-59.png)](https://postimg.cc/mzmGLF7V)



My Name Phonratichai Wairotchanaphuttha :)

# About This Respository

  * [simple counting helmetwearing program](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/tree/master/simeple_counting_helmetwearing)
  * [automatic detection for bikers with no helmet using deep learning paper](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/Automatic-Detector-for-Bikers-with-no-Helmet-using-Deep-Learning.pdf)
  * [Detection and Classification Vehicle paper th](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/Detection-and-Classification-Vehicle-PaperTH.pdf)
  * [Couting people with deeplearning program (NSC20)](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/NSC%E0%B8%84%E0%B8%A3%E0%B8%B1%E0%B9%89%E0%B8%87%E0%B8%97%E0%B8%B5%E0%B9%8820.pdf)

## simple_counting_helmetwearing 

this is process from object detection by tensorflow i create trainning data from capture image around my university
and have 2 label people_helmet_wearing and people_not_wearing_helmet
read documents -> [Document](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/simeple_counting_helmetwearing/file1.txt)

### program can recive input Video image and realtime with CCTV camera


example 

### people_helmet_wearing
[![1.jpg](https://i.postimg.cc/HLYQRSnm/1.jpg)](https://postimg.cc/HrNc79bv)

### people_not_wearing_helmet
[![2.jpg](https://i.postimg.cc/QtCFvkkR/2.jpg)](https://postimg.cc/ZBkYdNQc)

and then use create SSD for object detection (this image is not result)

[![Screen-Shot-2563-08-30-at-13-56-36.png](https://i.postimg.cc/ZqbrwN9d/Screen-Shot-2563-08-30-at-13-56-36.png)](https://postimg.cc/87YJ5jmp)

### flow processing program
[![Screen-Shot-2563-08-30-at-14-02-26.png](https://i.postimg.cc/qqVHTpH2/Screen-Shot-2563-08-30-at-14-02-26.png)](https://postimg.cc/9RpnP5Mf)


## automatic detection bikers with no helmet using deep learning paper
### Abstract
Abstract— The success of digital image pattern recognition and feature extraction using a Convolutional Neural Network (CNN) or Deep Learning was recently acknowledged over the years. Researchers have applied these techniques to many problems including traffic offense detection in video surveillance, especially for the motorcycle riders who are not wearing a helmet. Several models of CNN were used to solve these kinds of problem but mostly required the image pre- processing step for extracting the Region of Interest (ROI) area in the image before applying CNN to classify helmet. In this paper, we proposed to apply another interesting method of deep learning called Single Shot MultiBox Detector (SSD) into helmet detection problem. This method is the state-of-the-art that is able to use only one single CNN network to detect the bounding box area of motorcycle and rider and then classify that biker is wearing or not wearing a helmet at the same time. The results of the experiment were surprisingly good. The classification accuracy of bikers not wearing a helmet was extremely high and the detection of the ROI of biker and motorcycle in the image can be done at the same time as the classification process.   [full paper](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/Automatic-Detector-for-Bikers-with-no-Helmet-using-Deep-Learning.pdf)

## Detection and Classification Vehicle paper
### Abstract
การตรวจจับและจาแนกยานพาหนะ (Vehicle Detection andClassification)จากระบบวิดีโอรักษาความปลอดภัยเป็น การประยุกต์ใช้งานเทคนิคทางวิทยาการคอมพิวเตอร์ท่ีมี ความสาคัญอย่างยิ่งในปัจจุบัน โดยในมหาวิทยาลัยราชภัฏเลยมี จานวนยานพาหนะท่ีเป็นรถจักรยานยนต์และทาผิดกฏจราจร เป็นจานวนมากจนเกินความสามารถที่พนักงานรักษาความ ปลอดภัยจะตรวจตราได้ครบถ้วน การใช้ระบบตรวจจับ อัตโนมัติจึงมีความจาเป็นที่จะช่วยให้การตรวจจับการกระทา ผิดกฏจราจรและการรั กษาความปลอดภัยมีประสิ ทธิ ภาพที่ดีขึน้ ในงานวิจัยนี้มีจุดประสงค์ที่จะนาเทคนิคท่ีน่าสนใจทาง วิทยาการคอมพิวเตอร์ท่ีเรียกว่า การเรียนรู้เชิงลึก (Deep Learning) มาประยุกต์ใช้ในการพัฒนาระบบอัตโนมัติสาหรับ ตรวจจับและแยกแยกรถจักรยานยนต์ออกจากยานพาหนะ ประเภทอ่ืนๆ เพื่อแก้ไขปัญหาดังกล่าวและนาข้อมูลท่ีได้ไปใช้ ในการตรวจจับการทาผิดกฏจราจรต่อไป [full paper](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/Detection-and-Classification-Vehicle-PaperTH.pdf)

## Couting people with deeplearning program (NSC20)
### บทคัดย่อ
ในปัจจุบันเทคโนโลยีได้เข้ามามีบทบาทกับชีวิตประจาวันของเรามากขึ้นอาทิเช่น สมาร์ทโฟน หรือ คอมพิวเตอร์โน๊ตบุ๊ค ได้มีการใช้กันอย่างแพร่หลายนอกจากคอมพิวเตอร์โน๊ตบุ๊คแล้วยังมีเทคโนโลยีอย่างอื่นที่เรา พบเห็นหรือได้ใช้ในชีวิตประจาวันอีกมากมาย ยกตัวอย่างเช่นกล้องวงจรปิด หรือ กล้อง cctv ที่ใช้ในการเฝ้าระวัง ผู้ไม่หวังดี, ผู้ร้ายหรือภัยอันตรายต่างๆ จึงได้มีการพัฒนาระบบประมวลผลภาพ (Image Processing) และ คอมพิวเตอร์วิทัศน์ (Computer Vision) ขึ้นมาเพื่อช่วยในการวิเคราะห์รูปร่างหรือลักษณะผู้ร้าย นอกจากนี้ยังได้มี การนาระบบประมวลผลภาพ (Image Processing) และ คอมพิวเตอร์วิทัศน์ (Computer Vision) เข้ามาใช้ใน ด้านอื่นๆ อีกมามาย
Image Processing และ Computer Vision เป็นการนาภาพมาประมวลผลหรือคิดคานวณด้วย คอมพิวเตอร์ เพื่อให้คอมพิวเตอร์สามารถเข้าใจสิ่งที่อยู่ในภาพถ่าย นอกจากเทคนิคในส่วนของ Computer Vision แล้วยังมีอีกเทคนิคหนึ่งที่กาลังเป็นที่นิยมในขณะนี้ซึ่งก็คือ“Deep learning”เป็นการทาให้เครื่องจักรสามารถ ทานายหรือสร้างองค์ความรู้ได้โดยผ่านกระบวนการ 3 ขั้นตอนดังนี้ 1.นาเข้าชุดข้อมูล 2.สร้างโมเดลของชุดข้อมูล 3.ใช้โมเดลในการทานายข้อมูลชุดใหม่ซึ่งในทีนี้เราจะใช้หลักการ Deep learning ในการให้คอมพิวเตอร์เรียนรู้ว่า ภาพหรือสิ่งไหนคือ”มนุษย์”และสิ่งไหนไม่ใช่จากนั้นเราก็จะใช้หลักการ Object Detection เพื่อทาการแยกแยะ วัตถุที่เราจับได้จากโปรแกรมแล้วสร้างอัลกอริทึมในการนับจานวนของบุคคลจากภาพวิดิโอให้ได้ใกล้เคียงกับความ จริงมากที่สุด
ดังนั้นโครงงานนี้ จึงมีวัตถุประสงค์เพื่อศึกษาพัฒนาเทคนิคทางด้าน Image Processing และ Machine learning ในส่วนของการ Deep learning เพื่อใช้ในการประยุกต์ใช้กับการวิเคราะห์ภาพหรือบุคคลจากวิดีโอ เพื่อที่จะสามารถเพิ่มประสิทธิภาพความแม่นยาจากการใช้งานกล้องวงจรปิดและอื่นๆ ในการนับจานวนคนเพื่อ การประยุกต์ใช้ในงานด้านต่างๆ ต่อไป [full paper](https://github.com/FillOverFlow/Artificial-Intelligence-Portfolio/blob/master/NSC%E0%B8%84%E0%B8%A3%E0%B8%B1%E0%B9%89%E0%B8%87%E0%B8%97%E0%B8%B5%E0%B9%8820.pdf)
