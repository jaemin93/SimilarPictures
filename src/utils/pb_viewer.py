import tensorflow as tf
from tensorflow.python.platform import gfile

def _main():
    pb_path = input('where is your *.pb(graph)?')
    log_path = input('where is your log?')
    pb_view(pb_path, log_path)

def pb_view(path1, path2):
    with tf.Session() as sess:
        model_filename = path1
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    LOGDIR = path2
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)

    train_writer.flush()
    train_writer.close()

if __name__ == "__main__":
    _main()