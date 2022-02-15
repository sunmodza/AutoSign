import pickle
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
while True:
    command = input().lower()
    if command == "show":
        show_data()
    elif command == "del":
        num = int(input("index : "))
        del_data(num)
        show_data()

    elif command == "add":
        word_name = input("word_name : ")
        word_range = int(input("word_range : "))
        sentences = Sentences()
        sentences.word = word_name

        for i in range(word_range):
            sentence = Stage(input("sentence : "))
            sentences.add_stage(sentence)

        print(sentences,sentences.word)

        obj = load_data()
        obj.append(sentences)

        save_data(obj)
        show_data()

self.brain = [Sentences(*[Stage("1-9-9-0-6-0")],word = "สัตว์"),
                          Sentences(*[Stage("0-9-17-0-1-0"),Stage("1-9-9-0-5-0"),Stage("3-9-11-0-5-0")],word = "น้องสาว"),
                          Sentences(*[Stage("0-9-17-0-18-0"),Stage("1-9-9-0-5-0"),Stage("3-9-11-0-5-0")],word = "น้องสาว"),
                          Sentences(*[Stage("3-3-17-18-5-5"),Stage("2-3-17-18-5-5"),Stage("3-3-17-18-5-5")],word = "โรงเรียน"),
                          Sentences(*[Stage("3-4-2-4-20-20")],word = "นม"),
                          Sentences(*[Stage("2-2-11-11-11-11")], word="ผม"),
                          Sentences(*[Stage("9-3-0-4-0-1"),Stage("9-3-0-6-0-1")],word = "ชอบ"),
                          Sentences(*[Stage("3-3-1-3-5-5"),Stage("0-0-9-9-5-5")],word = "ใหญ่"),
                          Sentences(*[Stage("1-9-9-0-10-0"),Stage("1-9-6-0-11-0")],word = "โชคดี")]