import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import warnings
import re
import pandas as pd
from stemmer import Stemmer
warnings.filterwarnings("ignore")
from tensorflow.contrib.tensorboard.plugins import projector
# 


#hyper parameters 
VOCAB_SIZE = 5000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1            # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization_nep'
SKIP_STEP = 5000


NUM_VISUALIZE = 5000

file_path = "data/finaldata.csv"

df = pd.read_csv(file_path)
df= df.dropna()

df['sentence'] = df['sentence'].map(lambda x: re.sub('([?०१२३४५६७८९–!\\:...,\'‘()”“./—\\\])', '', x))


iwords = []
for sentence in df["sentence"]:
        try:
                iwords.extend(sentence.split())
        except:
                print(sentence, type(sentence))
print(len(iwords))


def subfunc(check_word, remove_words):
    if not list(set(list(check_word)[-len(remove_words):])- set(remove_words)):
      return check_word[:-len(remove_words)]
    else:
      return check_word


def check_case(check_word):
  remove_word_list = ["द्वारा", "बाट", "देखि", "लाई", "निम्ति", "मा", "को", "ले", "हरु"]
  return [subfunc(check_word, remove_words) if len(list(check_word[:-len(remove_words)])) >=3 else check_word for remove_words in remove_word_list][0]



st = Stemmer()
words = [st.stem(case) for case in iwords]


print(words[:100])



file = open(os.path.join(VISUAL_FLD, "vocab_nep.tsv"), "w")
from collections import Counter
dictionary = dict()
count = [('UNK', -1)]
index = 0
count.extend(Counter(words).most_common(VOCAB_SIZE - 1))
for word, _ in count:
  dictionary[word] = index
  index += 1
  file.write(word + '\n')
index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
file.close()




index_words =[dictionary[word] if word in dictionary else 0 for word in words]


print(len(words),len(index_words), len(dictionary))



del words
# del words , since google collab

#based on skip gram



import random
# def generate_sample(index_words , context_window_size):
#   #"made according to skip gram , each target context pair is treated as new data"
#   for index, center in enumerate(index_words):
#       #"center is index from dictionary and we need index to calculate index words"
#       context = random.randint(1,context_window_size)
#       # context is random , since context_window_size is 1 , it is always 1
#       # before the center words
#       for target in index_words[max(0, index-context):index]:
#         yield center , target
#       # after the center words
#       for target in index_words[index+1:index+1+context]:
#         yield center , target

def generate_sample(index_words , context_window_size):
    for key, row in df.iterrows():
        sentence = row['sentence'].split()
        sentence_limit = len(sentence)
        context = random.randint(1,context_window_size)
        for index in range(sentence_limit):
            if index == sentence_limit:
                pass
            else:
                for target in sentence[max(0,index-context):index]:
                        try:
                                yield dictionary[st.stem(sentence[index])], dictionary[st.stem(target)]
                        except KeyError:
                                pass
                for target in sentence[index+1:index+1+context]:
                        try:
                                yield dictionary[st.stem(sentence[index])], dictionary[st.stem(target)]
                        except KeyError:
                                pass

simple_gen = generate_sample(index_words, context_window_size = SKIP_WINDOW)

# data = next(simple_gen)
# print(index_dictionary[data[0]], index_dictionary[data[1]])
# data = next(simple_gen)
# print(index_dictionary[data[0]], index_dictionary[data[1]])
# data = next(simple_gen)
# print(index_dictionary[data[0]], index_dictionary[data[1]])




def batch_gen():
  simple_gen = generate_sample(index_words, context_window_size= SKIP_WINDOW)
  while True:
    center_batch= np.zeros(BATCH_SIZE, dtype= np.int32)
    target_batch= np.zeros([BATCH_SIZE, 1])
#     print(center_batch.shape, target_batch.shape)
    for index in range(BATCH_SIZE):
      center_batch[index], target_batch[index] = next(simple_gen)
    yield center_batch, target_batch

# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))
# print(next(batch_gen()))

print("all necessary task is done")


class word2Vec:
  def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate, dataset):
    self.vocab_size = vocab_size
    self.num_sampled = num_sampled
    self.embed_size = embed_size
    self.batch_size = batch_size
    self.num_sampled= num_sampled
    self.learning_rate= learning_rate
    self.dataset = dataset
    self.skip_step = SKIP_STEP
    self.global_step = tf.get_variable('global_step',
                                       initializer=tf.constant(0),
                                       trainable=False)
    
  def _import_data(self):
    #initialize data
    with tf.name_scope("data"):
      self.iterator = self.dataset.make_initializable_iterator()
      self.center_words, self.target_words = self.iterator.get_next()  
    
  def _create_embedding(self):
    #embedding setup
    with tf.name_scope("embed"):
      self.embed_matrix = tf.get_variable("embed_matrix",
                                          shape=[self.vocab_size,self.embed_size],
                                          initializer=tf.random_uniform_initializer())
      #compute forward pass of word2vec with NCE loss
      self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words,
                                     name="embedding")

  
  def _create_loss(self):
    #define loss
    with tf.name_scope("loss"):
      nce_weights = tf.get_variable("nce_weights",
                                         shape=[self.vocab_size,self.embed_size],
                                         initializer=
                                         tf.truncated_normal_initializer(
                                             stddev=1.0/(self.embed_size ** 0.5)))
      
      nce_biases = tf.get_variable("nce_biases",
                                        initializer= tf.zeros([self.vocab_size]))

      self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                               biases=nce_biases,
                                               labels=self.target_words,
                                               inputs=self.embed,
                                               num_sampled=self.num_sampled,
                                               num_classes=self.vocab_size),
                                name="loss")
   
      
  def _create_optimizer(self):
    #define optimizer 
    with tf.name_scope("gradients"):
      self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
          global_step=self.global_step)
    
    
  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar('loss', self.loss)
      tf.summary.histogram("histogram_loss", self.loss)
      self.summary_op = tf.summary.merge_all()
  
  def build_graph(self):
    #phase1
    #step 1 create dataset and samples
    self._import_data()
    #step 2 create embedding matrix, and inference ( compute forward path )
    self._create_embedding()
    #step 3 loss function
    self._create_loss()
    #step 4 optimizer and train instance
    self._create_optimizer()
    self._create_summaries()

  def main_process(self, num_tain_steps):
    #phase 2
    saver = tf.train.Saver()
    
    initial_step = 0
    
    with tf.Session() as sess:
      sess.run(self.iterator.initializer)
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter('graphs/word2vec_nep', sess.graph)
      ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints_nep/checkpoint'))
      
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      
      total_loss = 0.0
      writer = tf.summary.FileWriter('graphs/word2Vec_nep', sess.graph)
      initial_step = self.global_step.eval()
      
      for index in range(initial_step, initial_step + NUM_TRAIN_STEPS):
        try:
          loss_batch, _, summary = sess.run([self.loss,self.optimizer, self.summary_op])
          writer.add_summary(summary, global_step=index)
          total_loss+=loss_batch
          
          if (index +1)% self.skip_step ==0:
            print('Average loss at step {}: {:5.1f}'.format(index, total_loss/self.skip_step))
            total_loss =0.0
            saver.save(sess, 'checkpoints_nep/nepwv',index)
            
        except tf.errors.OutOfRangeError:
          sess.run(self.iterator.initializer)
      writer.close()
      
  def visualize(self,visual_fld, num_visualize):
    words = open(os.path.join(visual_fld, 'vocab_nep.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints_nep/checkpoint'))

      # if that checkpoint exists, restore from checkpoint
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

      final_embed_matrix = sess.run(self.embed_matrix)
    
      embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding')
      sess.run(embedding_var.initializer)
      print(embedding_var.shape)

      config = projector.ProjectorConfig()
      summary_writer = tf.summary.FileWriter(visual_fld)

      # add embedding to the config file
      embedding = config.embeddings.add()
      embedding.tensor_name = embedding_var.name

      # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
      embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'

      # saves a configuration file that TensorBoard will read during startup.
      projector.visualize_embeddings(summary_writer, config)
      saver_embed = tf.train.Saver([embedding_var])
      saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)



def data_generator():
  yield from batch_gen()


def main__():
  # tf.reset_default_graph()
  dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32, tf.int32),
                                           (tf.TensorShape([BATCH_SIZE]),
                                           tf.TensorShape([BATCH_SIZE,1])))
  
#   vocab_size, embed_size, batch_size, num_sampled, learning_rate, dataset
  W2V = word2Vec(vocab_size=VOCAB_SIZE,
                 embed_size=EMBED_SIZE,
                 batch_size=BATCH_SIZE,
                 num_sampled=NUM_SAMPLED,
                 learning_rate=LEARNING_RATE,
                 dataset=dataset)
  W2V.build_graph()
  W2V.main_process(NUM_TRAIN_STEPS)
  W2V.visualize(visual_fld=VISUAL_FLD,num_visualize=NUM_VISUALIZE)


if __name__ == "__main__":
	main__()
