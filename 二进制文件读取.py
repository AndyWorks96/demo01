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
        image_batch,label_batch=tf.train.batch([image_reshpae,label_cast],batch_size=1,num_threads=1,capacity=3)
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
    image_batch, label_batch=cr.read_and_decode(file_list)
    with tf.Session() as sess:
        # 创建线程协调器
        coord=tf.train.Coordinator()
        # 开启子线程去读取数据
        # 返回子线程实例
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        # 获取样本数据去训练
        print(sess.run([image_batch,label_batch]))

        # 关闭子线程，回收
        coord.request_stop()
        coord.join(threads)
