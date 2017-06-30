#

import tensorflow as tf

# Variableを作成して0で初期化
state = tf.Variable(0, name="counter")

# stateに1を足していくopの作成
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 全てのVariableを初期化するopの作成。（これがないと初期化されない）
init_op = tf.initialize_all_variables()

# sessionの実行
with tf.Session() as sess:
  # 初期化を行う
  sess.run(init_op)
  # stateの初期値を表示
  print(sess.run(state))
  # updateを実行してstateを表示する
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

# output:

# 0
# 1
# 2
# 3
