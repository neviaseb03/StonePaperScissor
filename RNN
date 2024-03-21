import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import random

moves = {'rock': 0, 'paper': 1, 'scissors': 2}
reverse_moves = {0: 'rock', 1: 'paper', 2: 'scissors'}


def generate_data(num_samples):
    sequences = []
    for _ in range(num_samples):
        sequences.append([random.randint(0, 2) for _ in range(3)])
    return sequences


data = generate_data(10000)
X = [to_categorical(seq[:-1], num_classes=3) for seq in data]
y = [seq[-1] for seq in data]

X = np.array(X)
y = to_categorical(y, num_classes=3)


model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(2, 3)),
    Dense(3, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32)


def make_prediction(history):
    history = to_categorical(history, num_classes=3).reshape(1, 2, 3)
    prediction = model.predict(history)
    return np.argmax(prediction)


def determine_winner(user_move, rnn_move):
    if user_move == rnn_move:
        return "It's a tie!"
    elif (user_move == 'rock' and rnn_move == 'scissors') or \
         (user_move == 'scissors' and rnn_move == 'paper') or \
         (user_move == 'paper' and rnn_move == 'rock'):
        return "You win!"
    else:
        return "RNN wins!"


def counter_repeated_moves(user_move, history):
    if len(set(history[-2:])) == 1:
        if user_move == 'rock':
            return 'paper'
        elif user_move == 'paper':
            return 'scissors'
        elif user_move == 'scissors':
            return 'rock'
    return None


history = [0, 0]
while True:
    user_move = input("Enter your move (rock, paper, scissors) or 'exit' to quit: ").lower()

    if user_move == 'exit':
        break

    if user_move not in moves:
        print("Invalid move. Please try again.")
        continue


    history.pop(0)
    history.append(moves[user_move])

    counter_move = counter_repeated_moves(user_move, history)
    if counter_move:
        rnn_move = counter_move
    else:
        rnn_move_index = make_prediction(history)
        rnn_move = reverse_moves[rnn_move_index]

    print(f"RNN's move: {rnn_move}")


    result = determine_winner(user_move, rnn_move)
    print(result)
