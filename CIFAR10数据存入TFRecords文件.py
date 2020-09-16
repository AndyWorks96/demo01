import tensorflow as tf
import os

class cifarRead(object):
    """
    二进制文件的读取
    """
    def __init__(self):
        # 定义图片的一些属性
        self.height=32
        self.width=32
        self.channel=3

        self.label_bytes=1
        self.image_bytes=self.height*self.width*self.channel
        self.bytes=self.image_bytes+self.label_bytes

    def read_and_decode(self,file_list):
        """
        读取二进制原始数据，并解码成张量
        :param file_list:
        :return:
        """
        # 1.构造文件队列
        file_queue=tf.train.string_input_producer(file_list)

        # 2.读取二进制数据
        reader=tf.FixedLengthRecordReader(self.bytes)
        key,value=reader.read(file_queue)
        print(value)

        # 3.进行二进制数据解码，decode——raw,(3073,),(3073,)=(1,)+(3072)
        # 为了训练方便，一般会把目标值和特征值分开
        label_image=tf.decode_raw(value,tf.uint8)
        print(label_image)
        # 使用tf.slice进行切片
        label=tf.slice(label_image,[0],[self.label_bytes])
        image=tf.slice(label_image,[self.label_bytes],[self.image_bytes])
        print(label,image)

        # 处理类型和图片数据的形状
        # 图片形状
        # reshape（3072，）---[channel,height,width]
        # transpose [channel,height,width]->[height,width,channel]

        label_cast=tf.cast(label,tf.int32)

        depth_major=tf.reshape(image,[self.channel,self.height,self.width])
        print(depth_major)
        image_reshpae=tf.transpose(depth_major,[1,2,0])
        print(image_reshpae)
        # 4.进行批处理，形状确定.
        image_batch,label_batch=tf.train.batch([image_reshpae,label_cast],batch_size=10,num_threads=1,capacity=10)
        return image_batch,label_batch

    def write_to_tfrecords(self, label_batch, image_batch):

        writer=tf.python_io.TFRecordWriter("./tmp/summary/cirfar.tfrecords")

        for i in range(10):
            # toString->tostring 报错
            image=image_batch[i].eval().tostring()
            label=label_batch[i].eval()[0]
            eample=tf.train.Example(features=tf.train.Features(feature={

                "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(eample.SerializeToString())
        writer.close()
        return None

    def read_tfrecords(self):
        """
        读取tfrecords文件
        :return:
        """
        # 1.构造tfrecords文件队列
        # 此处由于没有加方括号导致，一直报错。
        file_queue=tf.train.string_input_producer(["./tmp/summary/cirfar.tfrecords"])
        # file_queue = tf.train.string_input_producer("./tmp/summary/cirfar.tfrecords")
        print(file_queue)
        # 2.构造tfrecords文件读取器，读取队列
        reader=tf.TFRecordReader()

        # 默认只读一个样本
        _, value = reader.read(file_queue)
        # tfrecords多了example的解析步骤
        feature=tf.parse_single_example(value,features={
            "image":tf.FixedLenFeature([],tf.string),
            "label":tf.FixedLenFeature([],tf.int64)
        })
        # 取出feature里面的特征值和目标值
        # 通过键值对获取
        label=feature["label"]
        image=feature["image"]

        # 3.解码操作
        # 对于image是一个bytes类型，所以需要decode_raw去解码成uint8张量
        image=tf.decode_raw(image,tf.uint8)

        # 对于Label:本身是一个int类型，不需要去解码
        # 处理image的形状
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])
        # 处理label的形状和类型
        label_cast=tf.cast(label, tf.int32)

        # 4.批处理操作
        image_batch,label_batch=tf.train.batch([image_reshape,label_cast],batch_size=10,num_threads=1,capacity=10)
        print(image_batch, label_batch)
        return image_batch,label_batch

if __name__ == '__main__':

    # 生成路径+文件名的路径
    file_name=os.listdir("./data/cifar10/cifar-10-batches-bin/")
    print(file_name)

    # 路径+名字的拼接
    file_list=[os.path.join("./data/cifar10/cifar-10-batches-bin/",file) for file in file_name if file[-3:]=="bin"]
    print(file_list)
    #实体化类
    cr=cifarRead()
    image_batch,label_batch = cr.read_tfrecords()
    with tf.Session() as sess:
        # 创建线程协调器
        coord=tf.train.Coordinator()
        # 开启子线程去读取数据
        # 返回子线程实例
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        # print(label_batch[0].eval()[0])
        # 存入数据
        # cr.write_to_tfrecords(label_batch, image_batch)
        # 获取样本数据去训练

        print(sess.run([image_batch, label_batch]))
        # 关闭子线程，回收
        coord.request_stop()
        coord.join(threads)
