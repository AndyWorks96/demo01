import tensorflow as tf
import  os

def picread(file_list):
    """
    狗图片读取，转换成数据张量
    :param file_list:
    :return:
    """
    #构造文件队列
    file_queue=tf.train.string_input_producer(file_list)

    # 构造一个图片读取器，去对垒当中读取数据
    # 返回reader实例，调用read方法读取内容，key，value
    reader=tf.WholeFileReader()
    key,value=reader.read(file_queue)
    print(value)

    # 对样本内容进行解码
    image=tf.image.decode_jpeg(value)
    print(image)

    # 处理图片大小，形状，经过该函数处理图片数据类型编程float类型
    resize_image=tf.image.resize_images(image, [200, 200])
    print(resize_image)

    # 设置固定形状，使用静态api修改
    resize_image.set_shape([200,200,3])
    print(resize_image)

    # 批处理图片数据
    # 每个样本的形状必须全部定义 否则报错
    image_batch=tf.train.batch([resize_image],batch_size=1,num_threads=1,capacity=100)
    print(image_batch)
    return image_batch


if __name__ == '__main__':
    file_name=os.listdir("./data/dog/")
    # 路径拼接
    file_list=[os.path.join("./data/dog/",file) for file in file_name]

    print(file_list)
    image_batch=picread(file_list)
    with tf.Session() as sess:

        # 创建线程回收的协调员
        coord=tf.train.Coordinator()

        # 需手动开启子线程进行批处理读取到队列操作
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        print(sess.run(image_batch))

        coord.request_stop()
        coord.join(threads)