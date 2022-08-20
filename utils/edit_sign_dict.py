# use to edit the hand_dictionary
# provide function to load show save del data


import pickle
import re

from hand_lib import Sentences,Stage
def load_data():
    with open("sign_dictionary.pkl", "rb") as file:
        obj = pickle.load(file)
    return obj

def show_data():
    obj = load_data()
    for i,v in enumerate(obj):
        print(i,v.sentence,v.word)

def save_data(obj):
    with open("sign_dictionary.pkl", "wb") as file:
        pickle.dump(obj, file)

def del_data(i):
    obj = load_data()
    obj.pop(i)
    save_data(obj)

#show_data()
if __name__ == "__main__":
    while True:
        command = input().lower()
        if command == "show":
            show_data()
        elif command == "del":
            num = int(input("index : "))
            del_data(num)
            show_data()

        elif command == "search":
            name = input()
            data = load_data()
            await_remove = []
            for i,word in enumerate(data):
                if re.search(f"^{name}",word.word):
                    await_remove.append(i)
            for i in sorted(await_remove,reverse=True):
                del data[i]

            for word in data:
                print(word.word,word.sentence)

        elif command == "add":
            word_name = input("word_name : ")
            word_range = int(input("word_range : "))
            sentences = Sentences()
            sentences.word = word_name

            for i in range(word_range):
                sentence = Stage(input("sentence : "))
                sentences.add_stage(sentence)

            # print(sentences,sentences.word)

            obj = load_data()
            obj.append(sentences)

            save_data(obj)
            show_data()
